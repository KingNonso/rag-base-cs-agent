import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import jwt
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from . import store
from .config import Settings, settings
from .logging_setup import configure_logging, request_id_var
from .providers import make_providers
from .service import answer, ingest, replace

configure_logging()
logger = logging.getLogger("rag_support.api")


class Auth:
    """Verifies the existing application's JWTs against a static key or a cached JWKS."""

    def __init__(self, config: Settings):
        self.config = config
        self.jwks = jwt.PyJWKClient(config.jwt_jwks_url, cache_keys=True, lifespan=300) if config.jwt_jwks_url and not config.jwt_public_key else None

    def verify(self, token: str) -> dict:
        c = self.config
        if not c.jwt_issuer or not c.jwt_audience or not (c.jwt_public_key or self.jwks):
            raise HTTPException(503, "JWT verification is not configured")
        try:
            key = c.jwt_public_key.replace("\\n", "\n") if c.jwt_public_key else self.jwks.get_signing_key_from_jwt(token).key
            claims = jwt.decode(token, key, algorithms=c.algorithms, audience=c.jwt_audience, issuer=c.jwt_issuer, options={"require": ["exp", "sub", "iss", "aud"]})
        except jwt.PyJWTError as exc:
            raise HTTPException(401, "Invalid token") from exc
        account_id, user_id = claims.get(c.account_claim), claims.get("sub")
        if not isinstance(account_id, str) or not account_id or not isinstance(user_id, str) or not user_id:
            raise HTTPException(401, "Invalid token")
        roles = claims.get(c.roles_claim, [])
        return {"account_id": account_id, "user_id": user_id, "roles": roles if isinstance(roles, list) else []}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config: Settings = app.state.settings
    store.init_db(config.database_url, config.embedding_dimensions, config.embedding_id)
    if not hasattr(app.state, "embedder"):
        app.state.embedder, app.state.generator, app.state.reranker = make_providers(config)
    app.state.auth = Auth(config)
    yield


def create_app(config: Settings = settings, providers=None) -> FastAPI:
    app = FastAPI(title="RAG Customer Support", lifespan=lifespan)
    app.state.settings = config
    if providers:
        app.state.embedder, app.state.generator, app.state.reranker = providers
    origins = [o.strip() for o in config.cors_origins.split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST", "PUT", "DELETE"], allow_headers=["Authorization", "Content-Type", "X-Request-ID"], expose_headers=["X-Request-ID"])

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        supplied = request.headers.get("x-request-id", "")
        try:
            request_id = str(uuid.UUID(supplied))
        except ValueError:
            request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        started = time.monotonic()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            logger.exception("unhandled error", extra={"event": {"event": "error", "route": request.url.path}})
            response = JSONResponse({"detail": "Internal error", "request_id": request_id}, status_code=500)
        route = request.scope.get("route")
        logger.info("request", extra={"event": {"event": "http", "method": request.method, "route": getattr(route, "path", request.url.path), "status": status, "elapsed_ms": int((time.monotonic() - started) * 1000)}})
        response.headers["X-Request-ID"] = request_id
        request_id_var.reset(token)
        return response

    def principal(request: Request, authorization: str = Header(default="")):
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "Bearer token required")
        return request.app.state.auth.verify(authorization[7:])

    def admin(claims: dict = Depends(principal)):
        if config.admin_role not in claims["roles"]:
            raise HTTPException(403, "Admin role required")
        return claims

    def db():
        with store.connection(config.database_url) as conn:
            yield conn

    async def read_upload(file: UploadFile) -> bytes:
        data = await file.read(config.max_upload_bytes + 1)
        if len(data) > config.max_upload_bytes:
            raise HTTPException(413, "File too large")
        if not data:
            raise HTTPException(422, "Empty file")
        return data

    def provider_error(exc: Exception, stage: str):
        logger.warning("provider failure", extra={"event": {"event": "provider_error", "stage": stage, "error_type": type(exc).__name__}})
        return HTTPException(503, "A model provider is unavailable; try again later")

    class ChatRequest(BaseModel):
        message: str = Field(min_length=1, max_length=4000)
        session_id: uuid.UUID | None = None

    class FeedbackRequest(BaseModel):
        request_id: uuid.UUID
        rating: Literal["up", "down"]
        reason: Literal["incorrect", "unsupported", "unhelpful", "should_have_answered", "other"] | None = None
        comment: str | None = Field(default=None, max_length=2000)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/chat")
    def chat(body: ChatRequest, request: Request, claims: dict = Depends(principal), conn=Depends(db)):
        request_id = uuid.UUID(request.state.request_id)
        state = request.app.state
        try:
            result, trace = answer(conn, state.embedder, state.generator, state.reranker, body.message, claims["account_id"], claims["user_id"], body.session_id, config, request_id)
        except PermissionError as exc:
            raise HTTPException(404, "Session not found") from exc
        except (RuntimeError, OSError, ValueError) as exc:
            conn.rollback()
            raise provider_error(exc, "retrieval") from exc
        logger.info("chat", extra={"event": {"event": "chat", "account_id": claims["account_id"], "session_id": result["session_id"], "abstained": result["abstained"], **trace}})
        return {**result, "request_id": str(request_id)}

    @app.get("/v1/sessions/{session_id}/messages")
    def get_session(session_id: uuid.UUID, claims: dict = Depends(principal), conn=Depends(db)):
        if not store.owned_session(conn, session_id, claims["account_id"], claims["user_id"]):
            raise HTTPException(404, "Session not found")
        rows = store.session_messages(conn, session_id)
        return {"session_id": str(session_id), "messages": [{"role": r[0], "content": r[1], "references": r[2] or [], "abstain_reason": r[3], "request_id": str(r[4]) if r[4] else None, "created_at": r[5].isoformat()} for r in rows]}

    @app.get("/v1/references/{chunk_id}")
    def get_reference(chunk_id: uuid.UUID, claims: dict = Depends(principal), conn=Depends(db)):
        ref = store.get_reference(conn, chunk_id, claims["account_id"])
        if not ref:
            raise HTTPException(404, "Reference not found")
        return ref

    @app.post("/v1/feedback")
    def feedback(body: FeedbackRequest, claims: dict = Depends(principal), conn=Depends(db)):
        row = conn.execute(
            "SELECT m.session_id FROM messages m JOIN sessions s ON s.id=m.session_id WHERE m.request_id=%s AND m.role='assistant' AND s.account_id=%s AND s.user_id=%s",
            (body.request_id, claims["account_id"], claims["user_id"]),
        ).fetchone()
        if not row:
            raise HTTPException(404, "Response not found")
        conn.execute(
            """INSERT INTO feedback (request_id,session_id,account_id,user_id,rating,reason,comment) VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (request_id) DO UPDATE SET rating=EXCLUDED.rating, reason=EXCLUDED.reason, comment=EXCLUDED.comment, created_at=now()""",
            (body.request_id, row[0], claims["account_id"], claims["user_id"], body.rating, body.reason, body.comment),
        )
        logger.info("feedback", extra={"event": {"event": "feedback", "response_request_id": str(body.request_id), "rating": body.rating, "reason": body.reason, "has_comment": bool(body.comment)}})
        return {"recorded": True}

    # Parsing, OCR and embedding are blocking; keep them off the event loop.
    def _ingest(embedder, name, data, account_id, title, version, user_id):
        with store.connection(config.database_url) as conn:
            return ingest(conn, embedder, name, data, account_id, config, title=title, version_label=version, created_by=user_id)

    def _replace(embedder, document_key, name, data, title, version, user_id):
        with store.connection(config.database_url) as conn:
            return replace(conn, embedder, document_key, name, data, config, title=title, version_label=version, created_by=user_id)

    @app.post("/v1/documents", status_code=201)
    async def upload_document(request: Request, file: UploadFile = File(...), account_id: str | None = Form(default=None), title: str | None = Form(default=None, max_length=300), version: str | None = Form(default=None, max_length=100), claims: dict = Depends(admin)):
        if account_id is not None and not account_id.strip():
            raise HTTPException(400, "account_id must be nonempty")
        data = await read_upload(file)
        state = request.app.state
        try:
            result = await run_in_threadpool(_ingest, state.embedder, file.filename or "unnamed", data, account_id, title, version, claims["user_id"])
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (RuntimeError, OSError) as exc:
            raise provider_error(exc, "ingestion") from exc
        logger.info("document created", extra={"event": {"event": "document_created", "document_key": result["document_key"], "chunks": result["chunks"], "warnings": len(result["warnings"]), "shared": account_id is None}})
        return result

    @app.put("/v1/documents/{document_key}")
    async def replace_document(document_key: uuid.UUID, request: Request, file: UploadFile = File(...), title: str | None = Form(default=None, max_length=300), version: str | None = Form(default=None, max_length=100), claims: dict = Depends(admin)):
        data = await read_upload(file)
        state = request.app.state
        try:
            result = await run_in_threadpool(_replace, state.embedder, document_key, file.filename or "unnamed", data, title, version, claims["user_id"])
        except LookupError as exc:
            raise HTTPException(404, "Document not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (RuntimeError, OSError) as exc:
            raise provider_error(exc, "ingestion") from exc
        logger.info("document replaced", extra={"event": {"event": "document_replaced", "document_key": result["document_key"], "version": result["version"], "chunks": result["chunks"], "warnings": len(result["warnings"])}})
        return result

    @app.delete("/v1/documents/{document_key}")
    def delete_document(document_key: uuid.UUID, claims: dict = Depends(admin), conn=Depends(db)):
        if not store.delete_document(conn, document_key):
            raise HTTPException(404, "Document not found")
        logger.info("document deleted", extra={"event": {"event": "document_deleted", "document_key": str(document_key)}})
        return {"deleted": True}

    @app.get("/v1/documents")
    def list_documents(account_id: str | None = None, include_shared: bool = True, limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0), claims: dict = Depends(admin), conn=Depends(db)):
        return {"documents": store.list_documents(conn, account_id, include_shared, limit, offset)}

    @app.get("/v1/documents/{document_key}")
    def get_document(document_key: uuid.UUID, claims: dict = Depends(admin), conn=Depends(db)):
        versions = store.get_document_versions(conn, document_key)
        if not versions:
            raise HTTPException(404, "Document not found")
        return {"document_key": str(document_key), "current": next((v for v in versions if v["status"] == "active"), None), "versions": versions}

    return app


app = create_app()
