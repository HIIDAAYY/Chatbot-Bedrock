# WhatsApp AI Customer Care Chatbot (Twilio + Amazon Bedrock)

Webhook WhatsApp berbasis Twilio yang memanfaatkan Amazon Bedrock untuk jawaban AI, DynamoDB untuk state sesi, dan AWS SAM untuk deployment. Untuk pengujian tanpa WhatsApp, repositori ini juga menyediakan endpoint Discord Interactions (/discord) untuk slash command sederhana.

## Arsitektur

```
Twilio WhatsApp → Amazon API Gateway (HTTP API) → AWS Lambda (Python 3.12)
        │                                              │
        │                                              ├─► Amazon Bedrock Runtime (LLM / opsional RAG)
        │                                              ├─► Amazon DynamoDB (session + audit)
        │                                              └─► Amazon CloudWatch Logs & metrics
        └──────────────────────────────────────────────┴─► Twilio WhatsApp outbound message
```

Mode uji via Discord (opsional):

```
Discord Slash Command → API Gateway /discord → Lambda → (ACK defer)
                                           └─► Lambda (async follow-up) → Bedrock → Discord webhook reply
```

Jika `KNOWLEDGE_BASE_ID` diisi, Lambda menjalankan Retrieve-And-Generate (RAG) via Knowledge Bases for Amazon Bedrock.

## Prasyarat

- Python 3.12
- AWS CLI dan kredensial dengan akses ke SAM/CloudFormation, DynamoDB, Bedrock, Secrets Manager
- Akun Twilio dengan akses WhatsApp (trial/production) dan nomor/sender WhatsApp
- SAM CLI (`pip install aws-sam-cli`)
 - Opsional Discord testing: akun Discord Developer + Application (Client ID, Public Key)

## Konfigurasi Rahasia (Twilio)

Simpan kredensial Twilio (Account SID + Auth Token) di AWS Secrets Manager:

```bash
aws secretsmanager create-secret \
  --name twilio-bot-secrets \
  --secret-string '{"TWILIO_ACCOUNT_SID":"ACxxx","TWILIO_AUTH_TOKEN":"xxx"}'
```

Set variabel `TWILIO_SECRET_NAME` sesuai nama secret Anda. Jangan commit file `.env` yang berisi rahasia.

## Variabel Lingkungan

Salin `.env.example` menjadi `.env`, lalu isi minimal:

- `AWS_REGION`: region AWS (mis. `us-east-1`)
- `TWILIO_SECRET_NAME`: nama secret Twilio di Secrets Manager
- `TWILIO_WHATSAPP_FROM`: nomor pengirim WhatsApp (format `whatsapp:+1...`) **atau** `TWILIO_MESSAGING_SERVICE_SID`
- `BEDROCK_MODEL_ID`: ID model Bedrock yang tersedia di region Anda, mis. `anthropic.claude-3-sonnet-20240229-v1:0`
- Opsional: `KNOWLEDGE_BASE_ID`, `BEDROCK_GUARDRAIL_ID`, `BEDROCK_GUARDRAIL_VER`
- Opsional: set `TWILIO_VALIDATE_SIGNATURE=false` saat pengembangan lokal agar tidak perlu tanda tangan

## Instalasi Lokal

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Aktifkan variabel lingkungan (gunakan `.env` via tooling atau `export` manual).

## Menjalankan Secara Lokal

Bangun artefak:

```bash
sam build
```

Jalankan API lokal:

```bash
sam local start-api --env-vars sam-local-env.json
```

Contoh `sam-local-env.json`:

```json
{
  "ChatbotFunction": {
    "APP_ENV": "dev",
    "AWS_REGION": "us-east-1",
    "TWILIO_SECRET_NAME": "twilio-bot-secrets",
    "TWILIO_WHATSAPP_FROM": "whatsapp:+14155238886",
    "DDB_TABLE": "chatbot-bedrock-sessions",
    "BEDROCK_MODEL_ID": "anthropic.claude-3-sonnet-20240229-v1:0",
    "TWILIO_VALIDATE_SIGNATURE": "false"
  }
}
```

Uji webhook dengan koleksi Postman (`postman_collection.json`) menggunakan body `x-www-form-urlencoded` seperti Twilio.

### Uji via Discord (opsional)

1. Buat Discord Application di https://discord.com/developers/applications
2. Catat `Public Key` dan `Application ID`; isi ke parameter `DiscordPublicKey` dan `DiscordAppId` saat `sam deploy` (atau set di `.env` jika lokal).
3. Tambahkan slash command di tab Commands: `chat` dengan satu string option bernama `q`.
4. Set Interactions Endpoint URL ke output `DiscordEndpoint` (contoh: `https://.../v1/discord`). Setelah disimpan, Discord akan mengirim PING dan Lambda akan merespons PONG.
5. Invite bot ke server Anda (OAuth2 URL Generator → scopes `bot applications.commands`, minimal permission `Send Messages`).
6. Di server, jalankan `/chat q: <pertanyaan>`; bot akan menampilkan balasan dari Bedrock. Balasan dikirim sebagai follow-up agar melewati batas 3 detik Discord.

## Testing

```bash
make install
make test
```

Tes meliputi:

- Method GET ditolak (405)
- Alur pesan teks inbound Twilio (mock Bedrock & Twilio send)
- Persistensi state DynamoDB (Moto)
- Stub klien Bedrock & RAG

## Deploy ke AWS

1. `sam build`
2. `sam deploy --guided`
   - Isi parameter: `TwilioSecretName`, `TwilioWhatsappFrom` (atau kosong bila pakai Messaging Service), `TwilioMessagingServiceSid` bila diperlukan, `BedrockModelId`
   - Opsional: `KnowledgeBaseId`, `BedrockGuardrailId`, `BedrockGuardrailVersion`, `TwilioValidateSignature` (default true)
3. Catat output `ApiEndpoint` (Twilio) dan `DiscordEndpoint` (opsional)
4. Konfigurasikan webhook di Twilio Console → WhatsApp:
   - Set “WHEN A MESSAGE COMES IN” ke `https://<ApiEndpoint>/webhook`
   - Twilio otomatis mengirim POST ke endpoint tersebut

## Observabilitas & Keamanan

- **CloudWatch Logs**: log group `/aws/lambda/<StackName>-handler` mencatat event `message_received`, `twilio_message_sent`, `twilio_send_error`, dsb. Contoh query Log Insights:
  ```sql
  fields @timestamp, @message
  | filter @message like /message_received|twilio_message_sent|twilio_send_error/
  | sort @timestamp desc
  | limit 50
  ```
  Tambahkan `fields intent := extra.intent, escalate := extra.escalate` untuk melihat intent dan eskalasi.
- **Alarm rekomendasi**:
  1. Metric filter `twilio_send_error` → SNS alarm jika muncul >0 dalam 5 menit.
  2. Metric filter `escalate=true` → alarm jika eskalasi >5 dalam 15 menit.
  3. API Gateway 5XX error → bawaan metric `5XXError`.
- **Keamanan operasional**:
  - Simpan kredensial hanya di Secrets Manager (`twilio-bot-secrets`), jangan di repo.
  - Production: aktifkan `TWILIO_VALIDATE_SIGNATURE=true` dan gunakan HTTPS webhook.
  - Rotasi `TWILIO_AUTH_TOKEN` rutin; update secret + `sam deploy`.
  - IAM role pada template sudah least-privilege; tambah boundary jika perlu.
  - Jika Lambda berada di VPC, pastikan ada NAT/Internet Gateway agar bisa menghubungi Twilio API.

## RAG & Guardrail

- Jika `KNOWLEDGE_BASE_ID` diisi, Lambda memanggil `RetrieveAndGenerate` dan mengevaluasi skor relevansi. Skor rendah → respon aman + eskalasi.
- Parameter guardrail (opsional) diteruskan ke Bedrock jika tersedia.

## Pinecone FAQ (opsional)

- Jika Knowledge Base Bedrock belum tersedia, isi variabel `PINECONE_*` untuk memakai Pinecone sebagai vector store FAQ.
- Skrip `scripts/push_faq_to_pinecone.py` memecah Markdown di `kb/` dan mengirim teksnya saja. Index Pinecone bertipe “integrated embedding” akan mengubah teks menjadi vektor secara otomatis.
- Variabel penting: `PINECONE_API_KEY`, `PINECONE_ENV`, `PINECONE_INDEX`, `PINECONE_TOP_K`, `PINECONE_SCORE_THRESHOLD`.
- Runtime Lambda menambahkan hasil pencarian Pinecone ke prompt Bedrock ketika konfigurasi tersedia.

## Eskalasi ke Human

Jawaban low-confidence atau intent `out_of_scope` menghasilkan template aman dan menandai `escalation=true` pada state. Tambahkan integrasi tiket sesuai kebutuhan bisnis.

## Batasan

- Model Bedrock memerlukan izin/quota sesuai akun
- `check_order_status` masih dummy
- Pastikan token Twilio valid; token trial perlu verifikasi nomor penerima
- Signature Twilio wajib di production (`TWILIO_VALIDATE_SIGNATURE=true` secara default)
Mode uji tanpa platform (paling cepat):

```
Browser → API Gateway /ui → (POST /chat JSON) → Lambda → Bedrock → jawaban JSON
```

Testing cepat:
- Buka output `TestUiUrl` di browser, ketik pesan, klik Kirim.
- Atau kirim langsung:
  - `curl -X POST $JsonChatUrl -H 'Content-Type: application/json' -d '{"text":"halo"}'`
  - Respon: `{ "answer": "...", "intent": "...", "escalate": false }`
