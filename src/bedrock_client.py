"""Wrapper around Amazon Bedrock inference and Knowledge Bases."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from botocore.exceptions import ClientError

import config
from schemas import GeneratedAnswer, SessionState
import vector_store

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Anda adalah asisten layanan pelanggan resmi. "
    "Jawab ringkas, berbasis fakta, Bahasa Indonesia, sertakan sumber internal jika ada. "
    "Jika tidak yakin, nyatakan ketidakpastian dan sarankan bantuan manusia. "
    "Jangan berhalusinasi atau membuat kebijakan baru."
)

def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError("S3 URI must start with s3://")
    path = uri[5:]
    if "/" not in path:
        return path, ""
    bucket, key = path.split("/", 1)
    return bucket, key


class BedrockClient:
    """High level Bedrock inference client with optional RAG support."""

    def __init__(
        self,
        region: str,
        model_id: str,
        kb_id: Optional[str] = None,
        guardrail_id: Optional[str] = None,
        guardrail_ver: Optional[int] = None,
    ):
        if not model_id:
            raise config.ConfigurationError("BEDROCK_MODEL_ID must be provided")

        self.region = region
        self.model_id = model_id
        self.kb_id = kb_id
        self.guardrail_id = guardrail_id
        self.guardrail_ver = guardrail_ver
        self.inference_profile_arn = config.get_settings().bedrock_inference_profile_arn
        self._runtime = config.get_bedrock_runtime_client()
        self._agent_runtime = (
            config.get_bedrock_agent_runtime_client() if kb_id else None
        )
        # Load inline FAQ (optional) for fallback when KB is not configured
        self._inline_kb_text: Optional[str] = None
        try:
            self._inline_kb_text = self._load_inline_kb_text()
        except Exception:
            # Non-fatal: if inline load fails, just ignore
            self._inline_kb_text = None

    def _session_summary(self, session: Optional[SessionState]) -> str:
        if not session:
            return "Riwayat sesi tidak tersedia."
        summary_parts = []
        if session.last_intent:
            summary_parts.append(f"Intent terakhir: {session.last_intent}.")
        if session.last_reply:
            summary_parts.append(f"Balasan terakhir: {session.last_reply[:150]}.")
        if session.escalation:
            summary_parts.append("Status eskalasi: true.")
        if not summary_parts:
            return "Riwayat sesi minim."
        return " ".join(summary_parts)

    def _compose_prompt(self, question: str, session: Optional[SessionState], rag_context: Optional[str] = None) -> str:
        prompt_lines = [
            f"Sesi: {self._session_summary(session)}",
        ]

        if self.kb_id:
            # Placeholder wajib agar Bedrock KB menerima template RAG
            prompt_lines.append("Konten hasil pencarian:\n$search_results$")
            if rag_context:
                prompt_lines.append(f"Konteks tambahan:\n{rag_context}")
        else:
            if rag_context:
                prompt_lines.append(f"Konten relevan:\n{rag_context}")
            elif self._inline_kb_text:
                # If no KB configured, provide inline FAQ context when available
                prompt_lines.append(f"Konten relevan (FAQ internal):\n{self._inline_kb_text}")
        prompt_lines.append(f"Pengguna: {question}")
        prompt_lines.append("Jawab dalam paragraf singkat, gunakan Bahasa Indonesia.")
        return "\n\n".join(prompt_lines)

    def _pinecone_context(self, question: str) -> Optional[str]:
        matches = vector_store.search_chunks(question)
        if not matches:
            return None

        formatted: list[str] = []
        for idx, match in enumerate(matches, start=1):
            text = str(match.get("text", "")).strip()
            if not text:
                continue
            score = match.get("score")
            if isinstance(score, (int, float)):
                formatted.append(f"{idx}. (skor {score:.2f}) {text}")
            else:
                formatted.append(f"{idx}. {text}")

        if not formatted:
            return None

        return "\n".join(formatted)

    def _compose_additional_context(self, question: str) -> Optional[str]:
        sections: list[str] = []

        pinecone_context = self._pinecone_context(question)
        if pinecone_context:
            sections.append(f"Hasil Pinecone:\n{pinecone_context}")

        if not self.kb_id and self._inline_kb_text:
            sections.append(f"FAQ internal:\n{self._inline_kb_text}")

        if not sections:
            return None

        return "\n\n".join(sections)

    def _load_inline_kb_text(self) -> Optional[str]:
        """Load FAQ text to be inlined into prompts (optional).

        Priority:
        1) Local path from env `FAQ_INLINE_PATH`
        2) S3 object from env `FAQ_INLINE_S3_URI` (format s3://bucket/key)
        Result is truncated to `FAQ_INLINE_MAX_CHARS`.
        """
        settings = config.get_settings()
        if self.kb_id:
            return None

        max_chars = max(2000, int(settings.faq_inline_max_chars or 18000))

        # Try local filesystem
        import os
        path = settings.faq_inline_path
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = f.read()
                return data[:max_chars]
            except Exception:
                pass

        # Try S3
        s3_uri = settings.faq_inline_s3_uri
        if s3_uri and s3_uri.startswith("s3://"):
            try:
                bucket, key = _parse_s3_uri(s3_uri)
                s3 = config.get_s3_client()
                obj = s3.get_object(Bucket=bucket, Key=key)
                body = obj["Body"].read().decode("utf-8", errors="ignore")
                return body[:max_chars]
            except Exception:
                return None
        return None

    def _invoke_model(self, prompt: str) -> Dict[str, Any]:
        payload: Dict[str, Any]
        if "claude" in self.model_id:
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "system": SYSTEM_PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
                "max_tokens": 400,
                "temperature": 0.2,
                "top_p": 0.9,
            }
        else:
            payload = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "temperature": 0.2,
                    "topP": 0.9,
                    "maxTokenCount": 400,
                },
                "system": SYSTEM_PROMPT,
            }

        invoke_kwargs: Dict[str, Any] = {
            "modelId": self.model_id,
            "accept": "application/json",
            "contentType": "application/json",
            "body": json.dumps(payload).encode("utf-8"),
        }
        if self.guardrail_id and self.guardrail_ver:
            invoke_kwargs["guardrailConfig"] = {
                "guardrailIdentifier": self.guardrail_id,
                "guardrailVersion": str(self.guardrail_ver),
            }
        if self.inference_profile_arn:
            invoke_kwargs["inferenceProfileArn"] = self.inference_profile_arn

        try:
            response = self._runtime.invoke_model(**invoke_kwargs)
        except ClientError as exc:
            logger.error("bedrock_invoke_model_error", extra={"error": str(exc)})
            raise

        body = response.get("body")
        if hasattr(body, "read"):
            data = body.read()
        else:
            data = body
        return json.loads(data)

    @staticmethod
    def _extract_text_from_response(model_response: Dict[str, Any]) -> str:
        if "outputText" in model_response:
            return model_response["outputText"]

        if "result" in model_response:
            return model_response["result"]

        if "content" in model_response and isinstance(model_response["content"], list):
            for block in model_response["content"]:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "")

        if "outputs" in model_response:
            outputs = model_response["outputs"]
            if outputs and isinstance(outputs, list):
                text_block = outputs[0].get("text")
                if text_block:
                    return text_block

        return ""

    def answer_plain(self, question: str, session_ctx: Optional[SessionState]) -> GeneratedAnswer:
        """Generate an answer using InvokeModel without Knowledge Base context."""
        context_block = self._compose_additional_context(question)
        prompt = self._compose_prompt(question, session_ctx, rag_context=context_block)
        response_payload = self._invoke_model(prompt)
        answer_text = self._extract_text_from_response(response_payload).strip()
        confidence = 0.65
        if response_payload.get("stop_reason") == "end_turn":
            confidence = 0.7
        return GeneratedAnswer(answer=answer_text, confidence=confidence)

    def answer_with_rag(self, question: str, session_ctx: Optional[SessionState]) -> GeneratedAnswer:
        """Generate an answer using Knowledge Base RetrieveAndGenerate."""
        if not self.kb_id or not self._agent_runtime:
            raise config.ConfigurationError("KNOWLEDGE_BASE_ID must be configured for RAG flow")

        context_block = self._compose_additional_context(question)
        prompt = self._compose_prompt(question, session_ctx, rag_context=context_block)
        retrieve_config: Dict[str, Any] = {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": self.kb_id,
                "modelArn": f"arn:aws:bedrock:{self.region}::foundation-model/{self.model_id}",
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {"numberOfResults": 4}
                },
                "generationConfiguration": {
                    "promptTemplate": {"textPromptTemplate": prompt}
                },
            },
        }

        request: Dict[str, Any] = {
            "input": {"text": question},
            "retrieveAndGenerateConfiguration": retrieve_config,
        }

        if self.guardrail_id and self.guardrail_ver:
            request["guardrailConfiguration"] = {
                "guardrailIdentifier": self.guardrail_id,
                "guardrailVersion": str(self.guardrail_ver),
            }

        try:
            response = self._agent_runtime.retrieve_and_generate(**request)
        except ClientError as exc:
            logger.error("bedrock_retrieve_and_generate_error", extra={"error": str(exc)})
            raise

        output = response.get("output", {})
        answer_text = output.get("text", "").strip()
        citations = response.get("citations", [])

        highest_score = 0.0
        for citation in citations:
            for source in citation.get("sources", []):
                score = source.get("score")
                if isinstance(score, (int, float)):
                    highest_score = max(highest_score, float(score))

        confidence = highest_score if highest_score > 0 else 0.75 if answer_text else 0.0
        return GeneratedAnswer(answer=answer_text, confidence=confidence, citations=citations)
