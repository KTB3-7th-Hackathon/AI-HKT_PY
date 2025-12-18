from __future__ import annotations

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


app = FastAPI(title="geminitest API", version="0.1.0")


class ReportRequest(BaseModel):
    url: str = Field(..., min_length=1)
    tag: str = Field(..., min_length=1)

    model_config = ConfigDict(populate_by_name=True)


class ReportResponse(BaseModel):
    report_text: str = Field(..., alias="reportText")
    sentences: List[str]
    weight: int
    tag: str

    model_config = ConfigDict(populate_by_name=True)


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/api/report")
def create_report(request: ReportRequest):
    """
    현재는 컨트롤러 인터페이스만 고정하기 위한 기본 구현입니다.
    실제 서비스 로직(유튜브 스크립트 추출 → RAG/가공)은 추후 서비스 레이어가 준비되면 연결합니다.
    """
    response = ReportResponse(
        report_text="",
        words=[],
        weight=0,
        tag=request.tag,
    )

    return response.model_dump(by_alias=True)
