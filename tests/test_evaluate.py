from rag_support.evaluate import render_markdown, score_case, summarize

KEYS = {"k1": "product.md", "k2": "policy.txt"}


def result(answer="Basic allows three agents.", abstained=False, refs=("k1",), reason=None):
    return {"answer": answer, "abstained": abstained, "abstain_reason": reason, "references": [{"document_key": k} for k in refs]}


def test_direct_answer_passes():
    case = {"expected_documents": ["product.md"], "key_facts": ["three"], "expected_abstain": False}
    m = score_case(case, result(), {"candidates": [{"document_key": "k2"}, {"document_key": "k1"}]}, KEYS)
    assert m["passed"] and m["first_relevant_rank"] == 2


def test_wrong_citation_and_missing_fact_fail():
    case = {"expected_documents": ["product.md"], "key_facts": ["twenty"], "expected_abstain": False}
    m = score_case(case, result(refs=("k2",)), {"candidates": [{"document_key": "k2"}]}, KEYS)
    assert set(m["failures"]) == {"retrieval_hit", "citation_support", "answer_correct"}


def test_expected_abstention():
    case = {"expected_documents": [], "key_facts": [], "expected_abstain": True}
    assert score_case(case, result(abstained=True, refs=(), reason="insufficient"), {"candidates": []}, KEYS)["passed"]
    m = score_case(case, result(), {"candidates": []}, KEYS)
    assert m["failures"] == ["abstention_correct"]


def test_judge_can_accept_paraphrase_and_reject_unsupported():
    case = {"expected_documents": ["product.md"], "key_facts": ["three"], "expected_abstain": False}
    m = score_case(case, result(answer="Basic includes 3 agents."), {"candidates": [{"document_key": "k1"}]}, KEYS, {"correct": True, "supported": False})
    assert m["answer_correct"] is True and m["citation_support"] is False


def test_summary_and_markdown():
    case = {"expected_documents": ["product.md"], "key_facts": ["three"], "expected_abstain": False}
    rows = [{"id": "q1", "category": "direct", "question": "Q?", "expected": "Three", "expected_abstain": False, "answer": "three", "references": [], "metrics": score_case(case, result(), {"candidates": [{"document_key": "k1"}]}, KEYS)}]
    summary = summarize(rows)
    assert summary["passed"] == 1 and summary["retrieval_hit"] == {"passed": 1, "applicable": 1}
    text = render_markdown(rows, summary, {"run_id": "r", "embedding_model": "e", "generation_model": "g", "rerank_provider": "lexical", "min_retrieval_score": 0.2})
    assert "| q1 | direct | ✓ | ✓ | ✓ | ✓ | pass |" in text
