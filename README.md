# Grounded — RAG Customer Support Assistant

A public FastAPI prototype that turns support documents into cited answers. It parses PDF, DOCX, TXT, Markdown, and HTML, stores embeddings in PostgreSQL with pgvector, checks account scope during retrieval, and abstains when evidence is weak. The included documents and questions are fictional fixtures.

**[Try the frontend](https://kingnonso.github.io/rag-base-cs-agent/)** · [API docs after starting the backend](http://localhost:8000/docs)

The GitHub Pages site is a static sample by default. GitHub Pages cannot host the Python API or keep provider keys. You can connect the site to a separately deployed backend with a test JWT; the token stays in the browser tab’s memory.

## What is included

| Component | Behavior |
| --- | --- |
| Ingestion | PDF text and scanned PDF OCR, DOCX, TXT, Markdown, HTML; paragraph chunks with source locations |
| Retrieval | Exact cosine search in pgvector, filtered to shared and current-account documents, followed by term-overlap reranking |
| Generation | OpenAI or Anthropic answer provider; OpenAI or local sentence-transformer embeddings |
| Reliability | Source references, minimum similarity gate, explicit abstention, account-owned sessions |
| Evaluation | Ten labeled fixture questions and a CLI that records answers for review |
| Frontend | GitHub Pages sample chat with an optional live backend connection |

## Run locally

1. Install Python 3.11+ and start PostgreSQL with pgvector: `docker compose up -d db`. The included Compose file exposes it on localhost port 5433.
2. Create a virtual environment and install the package: `python -m venv .venv && .venv/bin/pip install -e '.[test]'`. Install Poppler (`pdftoppm`) and Tesseract if you need scanned PDF OCR. For local embeddings, also install `.[local]`.
3. Copy `.env.example` to `.env`. Set an OpenAI key for default embeddings and generation. Set `JWT_ISSUER`, `JWT_AUDIENCE`, and `JWT_JWKS_URL` for your app. No credentials are included in this repository.
4. Start the API: `.venv/bin/uvicorn rag_support.api:app --reload`. Open `http://localhost:8000/docs` for endpoint schemas.
5. Run the fixture evaluation: `.venv/bin/rag-evaluate --documents fixtures/documents --questions fixtures/questions.json`. This CLI operates locally without a JWT; the HTTP API always requires one.

For Anthropic answers, set `GENERATION_PROVIDER=anthropic`, a supported `GENERATION_MODEL`, and `ANTHROPIC_API_KEY`. For local embeddings, install `.[local]`, set `EMBEDDING_PROVIDER=local`, `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`, and `EMBEDDING_DIMENSIONS=384` before the first ingestion. Changing the embedding model or dimensions requires rebuilding the index in a new database. The app creates its tables at startup.

## API

- `POST /v1/documents`: multipart `file` and optional `account_id`; omitted `account_id` creates a shared document. Requires JWT `roles` including `knowledge_admin`.
- `DELETE /v1/documents/{id}`: deactivates a document. Requires admin role.
- `POST /v1/chat`: JSON `{"message":"...","session_id":null}`. Returns answer, references, abstained, session ID, and request ID.

JWTs require `sub`, `exp`, and `account_id` claims and matching issuer/audience. The API verifies RS256 or ES256 signatures through the configured JWKS endpoint. Admins can publish shared or account-specific documents; restrict this role to trusted staff in the existing application.

## GitHub Pages frontend

The site lives in [`docs/index.html`](docs/index.html) and deploys through [the Pages workflow](.github/workflows/pages.yml). In sample mode, it uses prepared responses from the fictional fixtures; it does **not** call an LLM. To use live mode, deploy the backend on an HTTPS host, set `CORS_ORIGINS=https://kingnonso.github.io` in its environment, and use the site's **Connect a deployed backend** panel with a test JWT. The browser sends requests directly to that host. Configure the real application's authentication and document access before exposing this to customers.

## Verification

Run `.venv/bin/pytest -q`. The suite checks parsing/chunking and abstention for missing evidence or invalid citations. A live PostgreSQL smoke check has also verified vector search, shared versus account-specific document access, citations, and session ownership. Run the evaluation CLI against real documents and labeled questions before using answers with customers.

## Retrieval and reliability

The parser retains PDF page numbers and DOCX/text document references. Chunks follow paragraph boundaries with overlap. Retrieval uses exact cosine search in pgvector with the account filter inside SQL, then blends semantic score with query-term overlap. The generator sees at most six excerpts and must return structured citation IDs. Missing evidence, low similarity, and malformed citations lead to a support handoff answer. The cutoff is a starting value; calibrate it on real labeled questions before customer rollout. Citations are validated for existence and account scope, but automatic entailment verification is not implemented.

The evaluation CLI prints expected labels, answers, and cited excerpts and saves results to PostgreSQL for manual review. Review correctness, support of each claim, and abstention separately. The fixtures do not establish real-world accuracy. Scanned PDFs need Poppler and Tesseract and should be reviewed for OCR quality. At larger scale, benchmark retrieval before adding an HNSW index; approximate search with account filters can reduce recall. This prototype does not automatically verify that every generated claim is entailed by its cited passage.
