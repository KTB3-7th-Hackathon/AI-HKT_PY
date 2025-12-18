import os
from youtube_transcript_api import YouTubeTranscriptApi
from google import genai
from google.genai import types

import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel


# Configuration
# Using the key from your previous files. Ideally move to environment variable.
# API_KEY = os.environ.get("GEMINI_API_KEY", "AA")


# 1. 초기화 (프로젝트 ID와 지역 설정)
# 지역은 보통 'asia-northeast3'(서울) 또는 'us-central1'을 사용합니다.
vertexai.init(project="strong-retina-481600-g2", location="us-central1")

# 2. 모델 로드 (해커톤은 속도가 빠른 flash 추천)
model = GenerativeModel("gemini-2.5-flash")
model_embedding = GenerativeModel("gemini-embedding-001")



# Initialize Client
# client = genai.Client(api_key=API_KEY)

def get_video_transcript(video_id):
    """Fetches Korean transcript from YouTube."""
    print(f"--- Fetching Transcript for {video_id} ---")
    try:
        ytt_api = YouTubeTranscriptApi()
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
    Refines the raw transcript using Gemini 2.5 Flash.
    Returns: {"refined": 정제된 텍스트, "summary": 3줄 요약}
    """
    print("\n--- Refining Script with Gemini ---")
    
    prompt = f'''
    SYSTEM:
    너는 뉴스 자막 편집 전문가다.

    INSTRUCTIONS:
    1. [정제] 원문 의미를 절대 바꾸지 마라. 요약하지 말고 전체 내용을 유지하라.
       - 자동 생성 자막의 오탈자, 중복, 잘린 문장을 복원하라
       - 줄바꿈은 문단 단위로 정리하라
       - 화자 기호(>>, - 등)는 제거하라
       - 추측이나 해석을 추가하지 마라

    2. [요약] 정제된 내용을 3줄 이내로 핵심만 요약하라.

    INPUT:
    {text}

    OUTPUT FORMAT (정확히 이 형식으로):
    [정제]
    (정제된 전체 텍스트)

    [요약]
    (3줄 이내 핵심 요약)
    '''

    try:
        response = model.generate_content(prompt)
        result_text = response.text
        
        # 파싱: [정제]와 [요약] 분리
        refined = ""
        summary = ""
        
        if "[정제]" in result_text and "[요약]" in result_text:
            parts = result_text.split("[요약]")
            refined = parts[0].replace("[정제]", "").strip()
            summary = parts[1].strip() if len(parts) > 1 else ""
        else:
            # 파싱 실패 시 전체를 refined로
            refined = result_text
            summary = "요약 생성 실패"
        
        return {"refined": refined, "summary": summary}


        #################################### 안해도됨 ########################################
        # Token counting (Checking verification/cost)
        # total_tokens = client.models.count_tokens(
        #     model="gemini-2.5-flash",
        #     contents=prompt
        # )
        # print(f"Token Count for Refinement: {total_tokens}")
        # print("Refinement Complete.")
        
        return response.text
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
    """정제된 텍스트 뭉치(Chunks)들을 벡터로 변환합니다."""
    print(f"\n--- Generating Embeddings for {len(chunks)} chunks ---")
    try:
        # Vertex AI의 임베딩 방식
        embeddings = model_embedding.get_embeddings(chunks)
        # 결과값에서 벡터 데이터만 추출
        return [embedding.values for embedding in embeddings]
    except Exception as e:
        print(f"임베딩 에러: {e}")
        return None

def main():
    # Youtube Video ID / 나중에 설정
    VIDEO_ID = "TCaDxE3wXsI"

    # 1. Fetch
    raw_script = get_video_transcript(VIDEO_ID)
    if not raw_script:
        return

    # 2. Refine
    refined_script = refine_script(raw_script)
    if not refined_script:
        return

    print("\n[Preview of Refined Script]")
    print(refined_script[:200] + "...\n")

    # 3. Embed
    # For embedding, we might want to split by lines or process the whole block.
    # Here we treat the whole refined text as one chunk, or split by lines/paragraphs.
    # Let's split by newline for demonstration if the result has paragraphs.
    chunks = [line for line in refined_script.split('\n') if line.strip()]
    
    if chunks:
        embeddings = create_embeddings(chunks)
        if embeddings:
            print(f"First embedding vector dimension: {len(embeddings[0].values)}")
    else:
        print("No content to embed.")

if __name__ == "__main__":
    main()
