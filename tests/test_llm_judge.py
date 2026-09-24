import httpx

from citecheck.judges import LLMJudge


def test_llm_judge_parses_answers():
    replies = iter(["Yes.", "no", "Maybe", ""])

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": next(replies)}}]})

    j = LLMJudge("m")
    j.http = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    assert j.score([("e", "c")] * 4) == [1.0, 0.0, 0.5, 0.5]
