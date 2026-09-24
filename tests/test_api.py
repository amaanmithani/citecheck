from fastapi.testclient import TestClient

from citecheck.api import create_app
from tests.test_metrics import DOCS, KeywordJudge


def test_check_endpoint():
    c = TestClient(create_app(judge=KeywordJudge(), threshold=0.5))
    r = c.post("/check", json={"answer": "Paris capital France [1]. Bananas purple [3].", "sources": DOCS})
    assert r.status_code == 200
    body = r.json()
    assert [s["supported"] for s in body["sentences"]] == [True, False]
    assert body["citation_recall"] == 0.5
    assert c.post("/check", json={"answer": "x", "sources": ["a" * 50_001]}).status_code == 413
    assert c.post("/check", json={"answer": "x", "sources": [], "threshold": 2}).status_code == 422
    assert c.get("/healthz").json() == {"status": "ok"}
    assert "citecheck" in c.get("/").text
