import hashlib
import math
import os
import re
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from rag_support.providers import LexicalReranker, terms

DIMENSIONS = 256


def _stem(term: str) -> str:
    return term[:-1] if len(term) > 3 and term.endswith("s") else term


class HashEmbedder:
    """Deterministic bag-of-words embedding: no network, stable across runs."""

    def __init__(self, dimensions: int = DIMENSIONS):
        self.dimensions = dimensions
        self.model_id = f"test:hash-{dimensions}"

    def embed(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for term in terms(text):
                vector[int(hashlib.md5(_stem(term).encode()).hexdigest(), 16) % self.dimensions] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


EXCERPT = re.compile(r"^\[(\d+)\] [^\n]*?\): (.*)$", re.M)
TIME = re.compile(r"\b\d{1,2}:\d{2}\b")


class ScriptedGenerator:
    """Cites excerpts sharing at least two question terms; flags disagreeing times as conflicts."""

    model_id = "test:scripted"

    def __init__(self):
        self.calls = []

    def answer(self, question, context, history):
        self.calls.append(question)
        wanted = {_stem(t) for t in terms(question)}
        scored = []
        for number, content in EXCERPT.findall(context):
            overlap = len(wanted & {_stem(t) for t in terms(content)})
            if overlap >= 2:
                scored.append((overlap, int(number), content))
        if not scored:
            return {"status": "insufficient", "answer": "", "citations": []}, "scripted-1"
        best = max(s[0] for s in scored)
        top = [s for s in scored if s[0] >= best - 1]
        times = {frozenset(TIME.findall(content)) for _, _, content in top if TIME.findall(content)}
        if len(times) > 1:
            return {"status": "conflicting", "answer": "", "citations": []}, "scripted-1"
        top.sort(key=lambda s: -s[0])
        return {"status": "answered", "answer": top[0][2], "citations": [n for _, n, _ in top[:2]]}, "scripted-1"

    def rewrite(self, question, history):
        previous = [line[6:] for line in history.splitlines() if line.startswith("user: ")]
        return f"{previous[-1]} {question}" if previous else question

    def judge(self, question, expected, answer, excerpts):
        return {"correct": True, "supported": True, "explanation": "scripted"}


@pytest.fixture
def providers():
    return HashEmbedder(), ScriptedGenerator(), LexicalReranker()


@pytest.fixture(scope="session")
def signing_key():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    return key, public


@pytest.fixture(scope="session")
def make_token(signing_key):
    key, _ = signing_key

    def make(account_id="account-a", sub="user-a", roles=(), **overrides):
        claims = {"iss": "https://app.test", "aud": "rag-support", "sub": sub, "account_id": account_id, "roles": roles if isinstance(roles, str) else list(roles), "exp": int(time.time()) + 600}
        claims.update(overrides)
        claims = {k: v for k, v in claims.items() if v is not None}
        return jwt.encode(claims, key, algorithm="RS256")

    return make


@pytest.fixture(scope="session")
def test_settings(signing_key):
    from rag_support.config import Settings

    return Settings(
        _env_file=None, database_url=os.environ.get("TEST_DATABASE_URL", "postgresql://invalid/none"), embedding_provider="test", embedding_model="hash-256",
        embedding_dimensions=DIMENSIONS, jwt_issuer="https://app.test", jwt_audience="rag-support", jwt_public_key=signing_key[1], jwt_jwks_url="",
        min_retrieval_score=0.1, cors_origins="",
    )
