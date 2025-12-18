import vertexai
from vertexai.generative_models import GenerativeModel
import os

# 프로젝트 정보
PROJECT_ID = "strong-retina-481600-g2"
LOCATION = "us-central1"

# 서비스 계정 키를 쓴다면 아래 주석 해제 (gcloud 설치 완료했으면 안 해도 됨)
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/key.json"

# Vertex AI 초기화
vertexai.init(project=PROJECT_ID, location=LOCATION)

# 모델 로드 (Gemini 2.0 Flash 사용 가능 시 모델명 확인 필요, 안전하게 1.5 추천)
# 구글 클라우드는 모델명이 'gemini-1.5-flash-002' 처럼 버전이 붙기도 합니다.
model = GenerativeModel("gemini-2.5-flash")

def test_gcp_gemini():
    try:
        response = model.generate_content("GCP Vertex AI 환경에서 보낸 메시지입니다. 잘 들리나요?")
        print(f"응답 결과: {response.text}")
    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == "__main__":
    test_gcp_gemini()