# WhatsApp AI Customer Care Chatbot (Amazon Bedrock)

Chatbot webhook for WhatsApp Cloud API that uses Amazon Bedrock for AI responses, DynamoDB for session state, and AWS SAM for deployment.

## Arsitektur

```
WhatsApp Cloud API ──► Amazon API Gateway (HTTP API) ──► AWS Lambda (Python 3.12)
      │                                                       │
      │                                                       ├─► Amazon Bedrock Runtime (LLM / RAG)
      │                                                       ├─► Amazon DynamoDB (session + audit)
      │                                                       ├─► Amazon CloudWatch Logs & metrics
      └───────────────────────────────────────────────────────┴─► WhatsApp Messages API (balasan)
```

Jika `KNOWLEDGE_BASE_ID` terisi, Lambda akan memakai Retrieve-And-Generate (RAG) via Knowledge Bases for Amazon Bedrock.

## Prasyarat

- Python 3.12
- AWS CLI + kredensial dengan akses SAM/CloudFormation, DynamoDB, Bedrock, Secrets Manager
- WhatsApp Cloud API: `phone_number_id`, akses token, app terhubung
- SAM CLI (`pip install aws-sam-cli`)

## Konfigurasi Rahasia

Simpan token WhatsApp di AWS Secrets Manager:

```
aws secretsmanager create-secret \
  --name wa-bot-secrets \
  --secret-string '{"WHATSAPP_ACCESS_TOKEN":"<token>","VERIFY_TOKEN":"<verification-string>"}'
```

Perbarui `WHATSAPP_SECRET_NAME` jika memakai nama lain. Jangan commit file `.env` yang berisi rahasia.

## Variabel Lingkungan

Salin `.env.example` ke `.env` dan isi nilai yang sesuai. Variabel penting:

- `AWS_REGION`: region AWS (mis. `us-east-1`)
- `WHATSAPP_GRAPH_VERSION`: versi Graph API, mis. `v20.0`
- `WHATSAPP_PHONE_NUMBER_ID`: ID dari WhatsApp Cloud API
- `BEDROCK_MODEL_ID`: ID model Bedrock yang tersedia di region mis. `anthropic.claude-3-sonnet-20240229-v1:0`
- `KNOWLEDGE_BASE_ID`: opsional untuk RAG (biarkan kosong jika tidak dipakai)
- `BEDROCK_GUARDRAIL_ID` / `BEDROCK_GUARDRAIL_VER`: opsional untuk mengaktifkan Guardrails

## Instalasi Lokal

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set variabel lingkungan (mis. melalui `export $(cat .env | xargs)` di Linux/macOS) atau gunakan mekanisme `.env` IDE.

## Menjalankan Secara Lokal

Bangun artefak:

```bash
sam build
```

Jalankan API lokal:

```bash
sam local start-api --env-vars sam-local-env.json
```

Buat file `sam-local-env.json` untuk menyuntik variabel runtime (contoh):

```json
{
  "ChatbotFunction": {
    "APP_ENV": "dev",
    "AWS_REGION": "us-east-1",
    "WHATSAPP_GRAPH_BASE": "https://graph.facebook.com",
    "WHATSAPP_GRAPH_VERSION": "v20.0",
    "WHATSAPP_PHONE_NUMBER_ID": "YOUR_PHONE_NUMBER_ID",
    "WHATSAPP_SECRET_NAME": "wa-bot-secrets",
    "DDB_TABLE": "chatbot-bedrock-sessions",
    "BEDROCK_MODEL_ID": "anthropic.claude-3-sonnet-20240229-v1:0"
  }
}
```

Uji endpoint menggunakan Postman atau koleksi `postman_collection.json`.

## Testing

```bash
make install
make test
```

Tes unit mencakup:

- Verifikasi webhook (GET)
- Alur pesan teks inbound (mock Bedrock & WhatsApp)
- Interaksi state DynamoDB (menggunakan Moto)
- Stub klien Bedrock

Pastikan waktu eksekusi handler di bawah 2,5 detik (dicek di tes).

## Deploy ke AWS

1. Jalankan `sam build`
2. Deploy dengan wizard:

   ```bash
   sam deploy --guided
   ```

   Isi parameter:
   - `WhatsAppGraphVersion`
   - `WhatsAppPhoneNumberId`
   - `BedrockModelId`
   - Opsional: `KnowledgeBaseId`, `BedrockGuardrailId`, `BedrockGuardrailVersion`

3. Catat output `ApiEndpoint`.
4. Konfigurasikan webhook di Meta Developers:

   - URL: `<ApiEndpoint>`
   - Verify token: sesuai dengan Secrets Manager

## Observabilitas

- Log terstruktur JSON di CloudWatch (`message_received`, `whatsapp_send_error`, dll.)
- Gunakan metric filter/CloudWatch Insights untuk menghitung eskalasi.
- Tambahkan alarm untuk error 5xx atau fallback guard.

## Penanganan RAG

Ketika `KNOWLEDGE_BASE_ID` terisi:

1. Lambda memanggil `RetrieveAndGenerate`.
2. Jika skor relevansi rendah, guard akan mengembalikan respons aman + eskalasi.
3. Metadata `citations` disimpan di state (bisa dikembangkan lebih lanjut).

Jika tidak terisi, sistem memakai inference generatif standar dengan prompt terkendali.

## Eskalasi ke Human

Respons low-confidence atau intent `out_of_scope` memakai template aman dan menandai `escalation=true` dalam session state. Tambahkan integrasi tiket di masa depan pada bagian TODO di kode.

## Batasan

- Model Bedrock bergantung pada izin dan quota akun.
- Order status masih dummy (`nlu.check_order_status`).
- Guardlist sederhana; pertimbangkan Guardrail resmi untuk proteksi tambahan.
- Pastikan secret WhatsApp direfresh sesuai masa berlaku token.

