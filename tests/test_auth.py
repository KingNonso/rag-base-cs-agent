import json
import logging
import time

import pytest
from fastapi import HTTPException

from rag_support.api import Auth
from rag_support.logging_setup import JsonFormatter, request_id_var


def test_valid_token_yields_verified_identity(test_settings, make_token):
    identity = Auth(test_settings).verify(make_token(roles=["knowledge_admin"]))
    assert identity == {"account_id": "account-a", "user_id": "user-a", "roles": ["knowledge_admin"]}


@pytest.mark.parametrize("overrides", [
    {"aud": "other"}, {"iss": "https://evil.test"}, {"exp": int(time.time()) - 10}, {"account_id": None}, {"account_id": ""}, {"sub": None}, {"account_id": 42},
])
def test_invalid_tokens_rejected(test_settings, make_token, overrides):
    with pytest.raises(HTTPException) as err:
        Auth(test_settings).verify(make_token(**overrides))
    assert err.value.status_code == 401


def test_wrong_signing_key_rejected(test_settings):
    from cryptography.hazmat.primitives.asymmetric import rsa
    import jwt
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = jwt.encode({"iss": "https://app.test", "aud": "rag-support", "sub": "u", "account_id": "a", "exp": int(time.time()) + 60}, other, algorithm="RS256")
    with pytest.raises(HTTPException) as err:
        Auth(test_settings).verify(token)
    assert err.value.status_code == 401


def test_hs256_token_not_accepted(test_settings):
    import jwt
    token = jwt.encode({"iss": "https://app.test", "aud": "rag-support", "sub": "u", "account_id": "a", "exp": int(time.time()) + 60}, "secret-key-that-is-long-enough-32b", algorithm="HS256")
    with pytest.raises(HTTPException):
        Auth(test_settings).verify(token)


def test_non_list_roles_ignored(test_settings, make_token):
    assert Auth(test_settings).verify(make_token(roles="knowledge_admin"))["roles"] == []


def test_unconfigured_auth_returns_503(test_settings, make_token):
    with pytest.raises(HTTPException) as err:
        Auth(test_settings.model_copy(update={"jwt_public_key": "", "jwt_jwks_url": ""})).verify(make_token())
    assert err.value.status_code == 503


def test_json_formatter_includes_request_id_and_event():
    record = logging.LogRecord("rag_support", logging.INFO, __file__, 1, "chat", None, None)
    record.event = {"top_score": 0.5}
    token = request_id_var.set("rid-1")
    try:
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert payload["request_id"] == "rid-1" and payload["top_score"] == 0.5 and payload["message"] == "chat"
