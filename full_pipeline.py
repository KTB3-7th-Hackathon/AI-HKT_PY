import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import re

# .env 파일 로드
load_dotenv()

# OpenAI API
from openai import OpenAI

# --- Configuration ---
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# OpenAI 클라이언트 초기화
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# Initialize Client
# client = genai.Client(api_key=API_KEY)

def get_video_transcript(url):
    """Fetches Korean transcript from YouTube."""
    print(f"--- Fetching Transcript for {url} ---")
    try:
        ytt_api = YouTubeTranscriptApi()

        #################################################
        # https://www.youtube.com/watch?v=YxmUIfr6HmU
        # asdf
        regex = r'(?:v=|\/|be\/|embed\/|shorts\/)([a-zA-Z0-9_-]{11})'

        match = re.search(regex, url)

        if match:
            video_id = match.group(1)
        else:
            # 이미 11자리 ID만 들어온 경우를 대비한 체크
            if len(url) == 11:
                video_id = url
        #################################################

        transcript_list = ytt_api.fetch(video_id, languages=['ko'])

        full_text = " ".join([snippet.text for snippet in transcript_list])

        text = full_text.strip()
        print(f"Successfully fetched {len(text)} characters.")
        return text
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None


def refine_script(text):
    """
    Refines the raw transcript using OpenAI GPT-4o-mini.
    Returns: {"refined": 정제된 텍스트, "summary": 3줄 요약}
    """
    print("\n--- Refining Script with GPT-4o-mini ---")

    system_prompt = """너는 뉴스 자막 편집 전문가다.

INSTRUCTIONS:
1. [정제] 원문 의미를 절대 바꾸지 마라. 요약하지 말고 전체 내용을 유지하라.
   - 자동 생성 자막의 오탈자, 중복, 잘린 문장을 복원하라
   - 줄바꿈은 문단 단위로 정리하라
   - 화자 기호(>>, - 등)는 제거하라
   - 추측이나 해석을 추가하지 마라

2. [요약] 정제된 내용을 3줄 이내로 핵심만 요약하라.

OUTPUT FORMAT (정확히 이 형식으로):
[정제]
(정제된 전체 텍스트)

[요약]
(3줄 이내 핵심 요약)"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        result_text = response.choices[0].message.content
        # 파싱 로직 보강
        refined = ""
        summary = ""

        # 좀 더 유연하게 찾기 위해 find 사용
        refined_idx = result_text.find("[정제]")
        summary_idx = result_text.find("[요약]")

        if refined_idx != -1 and summary_idx != -1:
            # [정제] 이후부터 [요약] 전까지
            refined = result_text[refined_idx + 4: summary_idx].strip()
            # [요약] 이후부터 끝까지
            summary = result_text[summary_idx + 4:].strip()
        else:
            # 형식이 틀렸을 경우 줄 단위로라도 시도
            lines = result_text.split('\n')
            refined = result_text
            summary = "요약 형식을 찾을 수 없음"
        print("refined : ",refined)

        return {"refined": refined, "summary": summary}

    except Exception as e:
        print(f"Error refining script: {e}")
        return None


# def create_embedding(text_list):
#     """Generates embeddings for a list of texts."""
#     print("\n--- Generating Embeddings ---")
#     try:
#         result = client.models.embed_content(
#             model="gemini-embedding-001",
#             contents=text_list,
#             # 모델 연산량 줄임 (768)
#             config=types.EmbedContentConfig(output_dimensionality=128)
#         )
#         print(f"Generated {len(result.embeddings)} embeddings.")
#         return result.embeddings
#     except Exception as e:
#         print(f"Error generating embedding: {e}")
#         return None

def create_embeddings(chunks):
    """정제된 텍스트 뭉치(Chunks)들을 벡터로 변환합니다. (OpenAI API 사용)"""
    print(f"\n--- Generating Embeddings for {len(chunks)} chunks ---")
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=chunks
        )
        # 결과값에서 벡터 데이터만 추출
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"임베딩 에러: {e}")
        return None

