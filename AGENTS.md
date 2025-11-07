# Repository Guidelines

## Project Structure & Modules
- `src/` hosts the Lambda service code: `app.py` is the handler, `bedrock_client.py` wraps Bedrock calls, `state.py` manages DynamoDB persistence, and channel adapters live in `whatsapp.py` and `discord_integration.py`. Shared validation models sit in `schemas.py`.
- `tests/` mirrors the runtime modules with pytest cases and fixtures in `conftest.py`; add new suites as `test_<feature>.py`.
- `events/` contains sample API Gateway payloads for quick local replay, while `kb/` stores knowledge-base Markdown seeded into Bedrock RAG flows.
- SAM build artefacts land in `.aws-sam/`; keep it ignored in commits.

## Build, Test, and Development Commands
- `make install` provisions `.venv` and installs both runtime (`src/requirements.txt`) and dev dependencies.
- `make test` (or `.venv/bin/pytest -q`) runs the unit suite with moto and responses stubs.
- `make build` wraps `sam build` for packaging the Lambda; `make local` runs `sam local start-api` to expose `/webhook` and `/discord` against `sam-local-env.json`.
- For quick manual checks, call `curl -X POST $JsonChatUrl -d '{"text":"hi"}' -H 'Content-Type: application/json'`.

## Coding Style & Naming Conventions
- Target Python 3.12, four-space indentation, and type hints (matches existing modules). Keep functions and modules in `snake_case`; classes (including Pydantic models) in `PascalCase`.
- Reuse helpers in `config.py` for logging setup and load secrets via environment variables, not inline literals.
- When adding handlers, follow the guardrail patterns in `guard.py` and centralise channel logic rather than branching in the entrypoint.

## Testing Guidelines
- Write pytest cases under `tests/` with descriptive `test_` names; colocate fixtures in `conftest.py` or dedicated helper modules.
- Mock AWS services with moto and HTTP calls with responses, as seen in `test_inbound_text_flow.py`.
- Cover both success and failure paths (Twilio signature, Bedrock fallbacks) before marking work ready for review.

## Commit & Pull Request Guidelines
- Use imperative, sentence-case commit messages with optional scopes (e.g. `Update config and bedrock client`, `Chore: trim pycache`). Squash noisy virtualenv changes before committing.
- PRs should summarise the change, list manual/automated test evidence (`make test`, `sam build`), link any tracking issues, and include screenshots or sample transcripts when UX behaviour changes.

## Security & Configuration Tips
- Duplicate `.env.example` for local secrets, but rely on AWS Secrets Manager (`TWILIO_SECRET_NAME`) in shared environments.
- Keep `TWILIO_VALIDATE_SIGNATURE` true outside local development and rotate tokens alongside secret updates.
- Review `template.yaml` IAM statements for least privilege whenever introducing new AWS integrations.
