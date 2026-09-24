"""HTTP API: POST /check scores an answer's citations; / serves a small page."""

from __future__ import annotations

import os
from functools import lru_cache
from importlib import resources

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .judges import CachedJudge, Judge, NLIJudge
from .metrics import check

DEFAULT_MODEL = os.environ.get("CITECHECK_MODEL", "cross-encoder/nli-deberta-v3-small")
# Threshold chosen on AttributionBench dev for the default judge (results/judges.json).
DEFAULT_THRESHOLD = float(os.environ.get("CITECHECK_THRESHOLD", "0.5"))


class CheckRequest(BaseModel):
    answer: str = Field(max_length=20_000)
    sources: list[str] = Field(max_length=50)
    threshold: float | None = Field(default=None, ge=0, le=1)


@lru_cache(maxsize=1)
def default_judge() -> Judge:
    return CachedJudge(NLIJudge(DEFAULT_MODEL))


def create_app(judge: Judge | None = None, threshold: float = DEFAULT_THRESHOLD) -> FastAPI:
    app = FastAPI(title="citecheck")

    @app.post("/check")
    def post_check(req: CheckRequest) -> dict[str, object]:
        if any(len(s) > 50_000 for s in req.sources):
            raise HTTPException(413, "each source must be at most 50,000 characters")
        j = judge or default_judge()
        return check(req.answer, req.sources, j, req.threshold if req.threshold is not None else threshold).to_dict()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return resources.files("citecheck").joinpath("static/index.html").read_text()

    return app


app = create_app()
