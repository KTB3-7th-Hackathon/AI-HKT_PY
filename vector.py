
import os
import pandas as pd
import faiss
import numpy as np
from google import genai
from google.genai import types
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

# --- 설정 (Configuration) ---
# 환경변수에서 API 키를 가져옵니다. 없을 경우 기본값을 사용하지만, 보안상 환경변수 사용을 권장합니다.
# API_KEY = os.environ.get("GEMINI_API_KEY", "AIz")
CSV_PATH = "testtest.csv"
INDEX_FILE = "faiss_index.bin"
DIMENSION = 768  # Gemini Embedding 001의 차원 수

# Gemini 클라이언트 초기화
# vector_embedding = GenerativeModel("gemini-embedding-001")


# 1. 초기화
PROJECT_ID = "strong-retina-481600-g2"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

# 2. 모델 로드 (GenerativeModel이 아니라 TextEmbeddingModel입니다!)
# 'text-embedding-004'는 현재 구글에서 가장 권장하는 최신 임베딩 모델입니다.
vector_embedding = TextEmbeddingModel.from_pretrained("text-embedding-004")



# --- 함수 정의 (Functions) ---

def load_data(csv_path):
    """
    CSV 파일을 불러와 DataFrame으로 반환합니다.
    """
    if not os.path.exists(csv_path):
        print(f"오류: {csv_path} 파일이 없습니다.")
        return None
    try:
        df = pd.read_csv(csv_path)
        print(f"{csv_path} 로드 완료. 데이터 개수: {len(df)}")
        return df
    except Exception as e:
        print(f"CSV 로드 중 오류 발생: {e}")
        return None

def get_embeddings(text_list):
    """
    텍스트 리스트를 입력받아 Gemini 임베딩 벡터 리스트를 반환합니다.
    """
    print(f"{len(text_list)}개의 텍스트 임베딩 생성 중...")
    try:
        # # 배치(Batch)로 요청하여 효율성을 높입니다.
        # result = vector_embedding.embed_content(
        #     contents=text_list,
        #     config=types.EmbedContentConfig(output_dimensionality=DIMENSION)
        # )
        # # 결과에서 벡터 값만 추출합니다.
        # embeddings = [embedding.values for embedding in result.embeddings]
        # return np.array(embeddings, dtype='float32') # FAISS는 float32를 사용합니다.

        # ✅ Vertex AI SDK의 메서드는 get_embeddings입니다.
        # task_type을 지정하면 검색(RETRIEVAL_DOCUMENT) 품질이 좋아집니다.
        inputs = [TextEmbeddingInput(text, task_type="RETRIEVAL_DOCUMENT") for text in text_list]
        result = vector_embedding.get_embeddings(inputs)

        # 결과에서 벡터 값(values)만 추출하여 numpy 배열로 변환
        embeddings = [embedding.values for embedding in result]
        return np.array(embeddings, dtype='float32')
    except Exception as e:
        print(f"임베딩 생성 실패: {e}")
        return None

def build_faiss_index(embeddings):
    """
    임베딩 벡터를 사용하여 FAISS 인덱스를 생성합니다.
    """
    # 벡터 차원 확인
    d = embeddings.shape[1] 
    if d != DIMENSION:
        print(f"경고: 임베딩 차원({d})이 설정된 차원({DIMENSION})과 다릅니다.")
    
    # IndexFlatIP: 코사인거리(L2) 기반의 가장 기본적인 인덱스 (정확도 높음, 속도 보통)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(d)
    
    # 벡터 추가
    index.add(embeddings)
    print(f"FAISS 인덱스 생성 완료. 저장된 벡터 수: {index.ntotal}")
    return index

def main():
    # 1. 데이터 로드 (Load Data)
    df = load_data(CSV_PATH)
    if df is None: return

    # 2. 임베딩 생성 (Generate Embeddings)
    # article.csv의 'content' 컬럼을 벡터화한다고 가정합니다.
    # 만약 다른 컬럼을 쓰고 싶다면 df['title'] 등으로 변경하세요.
    texts = df['content'].tolist()
    embeddings = get_embeddings(texts)
    
    if embeddings is None: return

    # 3. FAISS 인덱스 구축 (Build Index)
    index = build_faiss_index(embeddings)

    # 4. 인덱스 저장 (Save Index - Optional)
    faiss.write_index(index, INDEX_FILE)
    print(f"인덱스 파일 저장됨: {INDEX_FILE}")

    # 5. 검색 테스트 (Search Test)
    query_text = "Python coding" # 검색할 쿼리
    print(f"\n--- 검색 테스트: '{query_text}' ---")
    
    # 쿼리 벡터 생성
    query_embedding = get_embeddings([query_text])
    
    if query_embedding is not None:
        faiss.normalize_L2(query_embedding)
        
        # 검색 (Top 4 가까운 문서 검색)
        k = 4
        distances, indices = index.search(query_embedding, k)
        
        print("\n[검색 결과]")
        for i in range(k):
            idx = indices[0][i]
            dist = distances[0][i]
            if idx < len(df):
                meta_data = df.iloc[idx]
                print(f"순위 {i+1}: (거리: {dist:.4f})")
                print(f" - 제목: {meta_data.get('title', 'N/A')}")
                print(f" - 내용: {meta_data.get('content', 'N/A')}\n")

if __name__ == "__main__":
    main()