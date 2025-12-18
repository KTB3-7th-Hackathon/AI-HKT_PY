from __future__ import annotations

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


app = FastAPI(title="geminitest API", version="0.1.0")


class ReportRequest(BaseModel):
    video_url: str = Field(..., alias="videoUrl", min_length=1)
    tag: str = Field(..., min_length=1)

    model_config = ConfigDict(populate_by_name=True)


class ReportResponse(BaseModel):
    report_text: str = Field(..., alias="reportText")
    words: List[str]
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
    # TODO: 서비스 코드가 합쳐지면 아래에 실제 호출 로직을 연결
    # 1) video_url에서 video_id 추출
    # 2) 한국어 스크립트 추출
    # 3) tag 기반 벡터 DB 조회 및 리포트/편향도 산출
    # 4) {reportText, words, weight, tag} 형태로 반환

    response = ReportResponse(
        report_text="",
        words=[],
        weight=0,
        tag=request.tag,
    )
    return response.model_dump(by_alias=True)
