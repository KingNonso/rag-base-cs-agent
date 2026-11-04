import logging
import time
import uuid
from contextlib import asynccontextmanager

import jwt
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import settings
from .providers import make_providers
from .service import answer, ingest
from .store import connection, init_db


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_support")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(settings.database_url, settings.embedding_dimensions, f"{settings.embedding_provider}:{settings.embedding_model}")
    app.state.embedder, app.state.generator = make_providers(settings)
    yield


app = FastAPI(title="RAG Customer Support", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()], allow_methods=["POST"], allow_headers=["Authorization", "Content-Type"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: uuid.UUID | None = None


def principal(authorization: str = Header(default="")):
    if not settings.jwt_issuer or not settings.jwt_audience or not settings.jwt_jwks_url:
        raise HTTPException(503, "JWT verification is not configured")
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Bearer token required")
    try:
        key = jwt.PyJWKClient(settings.jwt_jwks_url).get_signing_key_from_jwt(authorization[7:]).key
        claims = jwt.decode(authorization[7:], key, algorithms=["RS256", "ES256"], audience=settings.jwt_audience, issuer=settings.jwt_issuer, options={"require": ["exp", "sub"]})
        if not isinstance(claims.get("account_id"), str) or not claims["account_id"]:
            raise ValueError("Missing account_id claim")
        return claims
    except (jwt.PyJWTError, ValueError) as exc:
        raise HTTPException(401, "Invalid token") from exc


def admin(claims: dict = Depends(principal)):
    roles = claims.get("roles", [])
    if not isinstance(roles, list) or settings.admin_role not in roles:
        raise HTTPException(403, "Admin role required")
    return claims


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat")
def chat(request: ChatRequest, claims: dict = Depends(principal)):
    started = time.monotonic()
    request_id = str(uuid.uuid4())
    try:
        with connection(settings.database_url) as conn:
            result = answer(conn, app.state.embedder, app.state.generator, request.message, claims["account_id"], claims["sub"], request.session_id, settings)
        logger.info("chat request_id=%s account_id=%s abstained=%s elapsed_ms=%d", request_id, claims["account_id"], result["abstained"], int((time.monotonic() - started) * 1000))
        return {**result, "request_id": request_id}
    except PermissionError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/v1/documents")
async def upload_document(file: UploadFile = File(...), account_id: str | None = Form(default=None), claims: dict = Depends(admin)):
    if account_id is not None and not account_id.strip():
        raise HTTPException(400, "account_id must be nonempty")
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "File too large")
    try:
        with connection(settings.database_url) as conn:
            return ingest(conn, app.state.embedder, file.filename or "unnamed", data, account_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.delete("/v1/documents/{document_id}")
def delete_document(document_id: uuid.UUID, claims: dict = Depends(admin)):
    with connection(settings.database_url) as conn:
        row = conn.execute("UPDATE documents SET active=false WHERE id=%s AND active=true RETURNING id", (document_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Document not found")
    return {"deleted": True}
