# Grounded — RAG Customer Support Assistant

A FastAPI service that turns support documents into cited answers. It parses PDF (including scanned PDFs through OCR), DOCX, TXT, Markdown, and HTML. Embeddings live in PostgreSQL with pgvector. Retrieval is limited to shared documents and the caller's own account, and the service abstains when evidence is missing or conflicting. The included documents and questions are fictional fixtures.

**[Try the frontend](https://kingnonso.github.io/rag-base-cs-agent/)** · [API docs after starting the backend](http://localhost:8000/docs) · [Prototype report](reports/prototype-report.md)

The GitHub Pages site is a static sample by default. GitHub Pages cannot host the Python API or keep provider keys. You can connect the site to a separately deployed backend with a test JWT; the token stays in the browser tab's memory.

## What is included

| Component | Behavior |
| --- | --- |
| Ingestion | PDF text and scanned-PDF OCR, DOCX, TXT, Markdown, HTML. Keeps title, version, page or heading path, and character offsets. Low-quality OCR pages are flagged but still indexed. |
| Chunking | Packs paragraphs into chunks of at most 1,300 characters with 180 characters of overlap. Chunks never cross a page or section. The embedded text is prefixed with the title and section. |
| Retrieval | Exact cosine search in pgvector with the account filter inside SQL, then reranking: lexical BM25 blend by default, or an optional cross-encoder. |
| Generation | OpenAI or Anthropic. The model must return `answered`, `insufficient`, or `conflicting` with citation numbers. |
| Reliability | Similarity gate. Abstains on insufficient or conflicting evidence, malformed output, or provider errors. Citations are checked for range and rechecked for access before they are returned. |
| Versions | Replacement creates a new version and retires the old one in the same transaction, only after the new version has parsed and embedded. Deletion purges chunks and keeps a tombstone. |
| Sessions | Owned by account and user. Recent turns rewrite follow-up questions, and evidence is retrieved fresh each turn. Retention is handled by `rag-purge`. |
| Observability | JSON logs with request IDs, timings, retrieval IDs and scores, model versions, and error types. No question, answer, or document text. Feedback endpoint. |
| Evaluation | Labeled questions are scored for retrieval hit, citation support, abstention, and answer correctness, with an optional LLM judge. Produces JSON and Markdown reports, stores runs in the database, and can gate CI with `--min-pass`. |

## Run locally

1. Install Python 3.11+ and start PostgreSQL with pgvector: `docker compose up -d db` (listens on localhost port 5433).
2. `python -m venv .venv && .venv/bin/pip install -e '.[test]'`. Install Poppler (`pdftoppm`) and Tesseract for scanned PDFs (`brew install poppler tesseract` or `apt install poppler-utils tesseract-ocr`). For local embeddings or the cross-encoder reranker, also install `.[local]`.
3. Copy `.env.example` to `.env`. Set an OpenAI key for the default embeddings and generation. Set `JWT_ISSUER`, `JWT_AUDIENCE`, and either `JWT_JWKS_URL` or `JWT_PUBLIC_KEY`. No credentials are included in this repository.
4. Start the API: `.venv/bin/uvicorn rag_support.api:app --reload`. Database migrations run at startup.
5. Run the evaluation: `.venv/bin/rag-evaluate --documents fixtures/documents --questions fixtures/questions.json --report reports/eval`.

For Anthropic answers, set `GENERATION_PROVIDER=anthropic`, `GENERATION_MODEL=claude-sonnet-5-5` (or another supported model), and `ANTHROPIC_API_KEY`. For local embeddings, set `EMBEDDING_PROVIDER=local`, `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`, and `EMBEDDING_DIMENSIONS=384`.

### Changing the embedding model

The database records which embedding model built the index, and the API refuses to start with a different one. To switch:

1. Pause document uploads.
2. Set the new `EMBEDDING_*` values.
3. Run `.venv/bin/rag-reindex`. It re-embeds every chunk into a new column and swaps it in one transaction; retrieval keeps using the old vectors until then.
4. Restart the API.

## API

Every endpoint except `/health` requires `Authorization: Bearer <JWT>`. Tokens must have a valid signature, `iss`, `aud`, `exp`, `sub`, and an account claim (`account_id` by default). Account and user IDs come only from the verified claims. Send `X-Request-ID` (a UUID) to correlate logs; it is echoed in the response.

**Customer endpoints**

- `POST /v1/chat` with `{"message": "...", "session_id": null}` returns `answer`, `references`, `abstained`, `abstain_reason`, `session_id`, and `request_id`. Each reference has `chunk_id`, `document_key`, `document` (title), `version_label`, `location`, `page`, `section`, `start_offset`, `end_offset`, and `excerpt`. Using another user's session returns 404.
- `GET /v1/sessions/{id}/messages`: your own session's history.
- `GET /v1/references/{chunk_id}`: the full cited passage. Returns 404 if it has been replaced, deleted, or belongs to another account.
- `POST /v1/feedback` with `{"request_id", "rating": "up"|"down", "reason"?, "comment"?}`, for your own responses.

**Admin endpoints** (JWT `roles` must include `ADMIN_ROLE`)

- `POST /v1/documents` (multipart `file`, plus optional `account_id`, `title`, `version`). Omitting `account_id` makes the document shared. Returns `document_key`, chunk count, and OCR `warnings`.
- `PUT /v1/documents/{document_key}` (multipart `file`, plus optional `title`, `version`): replace with a new version. If the new file fails to parse or embed, the old version stays live.
- `DELETE /v1/documents/{document_key}`: remove from retrieval and purge chunks.
- `GET /v1/documents?account_id=&include_shared=` and `GET /v1/documents/{document_key}`: metadata, version history, and OCR warnings.

Upload and replace run synchronously; a long scanned PDF can take a while to OCR. Restrict the admin role to trusted staff in the existing application.

## Evaluation

`fixtures/questions.json` lists questions with `id`, `category` (`direct`, `multi_passage`, `ambiguous`, `absent`, `conflicting`, `followup`), `expected_documents`, `key_facts`, `expected_abstain`, an optional `expected` reference answer, and optional `history`. Each run:

1. Ingests the folder into a throwaway account.
2. Scores every question:
   - **retrieval hit:** an expected document is among the retrieved candidates.
   - **citation support:** the answer cites only expected documents.
   - **abstention correctness.**
   - **answer correctness:** all key facts appear in the answer.
3. Writes `--report` files, saves the run with its configuration and git SHA, and deletes the documents.

`--judge` also asks the generation model whether the answer is correct and supported by the cited excerpts. `--min-pass 0.9` exits nonzero below a 90% pass rate. Run it whenever parsing, retrieval, prompts, or models change, and with the business's own documents and questions before rollout.

## Operations

- `rag-purge [--days N]` deletes sessions inactive for longer than `SESSION_RETENTION_DAYS`. Run it on a schedule.
- Logs are JSON on stdout. They contain request IDs, account and session IDs, timings, retrieved chunk IDs and scores, the abstain reason, embedding, generation and rerank model identifiers, and error types. They never contain the question, answer, feedback comment, or document text, and a test enforces this. Question and answer text is stored only in the `messages` table, for session history.

## Verification

- `.venv/bin/pytest -q` runs unit tests for parsing, chunk offsets, OCR flags, abstention paths, the reference recheck, JWT validation, and eval scoring.
- With a database, `TEST_DATABASE_URL=postgresql://rag:rag@localhost:5433/rag_test .venv/bin/pytest -q` also runs the acceptance tests. They use deterministic offline providers and check:
  - upload, replace, and delete, including scanned PDFs, change later answers;
  - references resolve;
  - unsupported and conflicting questions abstain;
  - accounts and sessions stay isolated, including supplied IDs and follow-ups;
  - logs contain no text;
  - reindexing works;
  - the evaluation report is produced.
- CI (`.github/workflows/ci.yml`) runs everything against pgvector with Tesseract installed.

## GitHub Pages frontend

The site lives in [`docs/index.html`](docs/index.html) and deploys through [the Pages workflow](.github/workflows/pages.yml). In sample mode, it shows prepared responses from the fictional fixtures and does **not** call an LLM.

For live mode:

1. Deploy the backend on an HTTPS host.
2. Set `CORS_ORIGINS=https://kingnonso.github.io` in its environment.
3. Use the site's **Connect a deployed backend** panel with a test JWT.

The browser sends requests directly to that host.
