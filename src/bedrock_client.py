"""Wrapper around Amazon Bedrock inference and Knowledge Bases."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from botocore.exceptions import ClientError

import config
from schemas import GeneratedAnswer, SessionState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Anda adalah asisten layanan pelanggan resmi. "
    "Jawab ringkas, berbasis fakta, Bahasa Indonesia, sertakan sumber internal jika ada. "
    "Jika tidak yakin, nyatakan ketidakpastian dan sarankan bantuan manusia. "
    "Jangan berhalusinasi atau membuat kebijakan baru."
)


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
        self._runtime = config.get_bedrock_runtime_client()
        self._agent_runtime = (
            config.get_bedrock_agent_runtime_client() if kb_id else None
        )

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
        if rag_context:
            prompt_lines.append(f"Konten relevan:\n{rag_context}")
        prompt_lines.append(f"Pengguna: {question}")
        prompt_lines.append("Jawab dalam paragraf singkat, gunakan Bahasa Indonesia.")
        return "\n\n".join(prompt_lines)

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
        prompt = self._compose_prompt(question, session_ctx)
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

        prompt = self._compose_prompt(question, session_ctx)
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

