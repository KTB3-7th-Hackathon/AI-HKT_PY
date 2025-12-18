import os
import pickle
import pandas as pd
import faiss
import numpy as np
import time
from typing import List, Dict, Optional

# OpenAI API
from openai import OpenAI

# --- 설정 (Configuration) ---
CSV_PATH = "testtest.csv"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"
DIMENSION = 1536  # text-embedding-3-small의 차원 수
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# OpenAI 클라이언트 초기화 (OPENAI_API_KEY 환경변수 필요)
client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))



class HKTVectorStore:
    def __init__(self, index_file: str = INDEX_FILE, metadata_file: str = METADATA_FILE):
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.index = None
        self.metadata: List[Dict] = []
        self.client = client  # OpenAI 클라이언트

    def load_data(self, csv_path: str) -> Optional[pd.DataFrame]:
        """CSV 파일을 로드합니다."""
        if not os.path.exists(csv_path):
            print(f"오류: {csv_path} 파일이 없습니다.")
            return None
        try:
            # skipinitialspace=True: 컬럼명이나 데이터 앞의 공백 무시
            df = pd.read_csv(csv_path, skipinitialspace=True)
            print(f"{csv_path} 로드 완료. 데이터 개수: {len(df)}")
            print(f"컬럼 목록: {df.columns.tolist()}")  # 디버깅용
            return df
        except Exception as e:
            print(f"CSV 로드 중 오류 발생: {e}")
            return None

    def chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 10) -> List[str]:
        """
        문장 단위로 똑똑하게 자르는 청킹 함수.
        단순히 글자 수로 자르지 않고, 마침표(.)나 줄바꿈(\n) 등을 고려합니다.
        """
        if not isinstance(text, str) or not text:
            return []

        # 1. 문장 단위로 분리 (간단한 규칙)
        # 실제로는 정규식 등으로 더 정교하게 할 수 있지만, 여기선 단순화
        raw_sentences = text.replace('\n', ' ').split('. ')
        sentences = [s.strip() + '.' for s in raw_sentences if s.strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # 현재 청크에 문장을 더해도 사이즈를 넘지 않으면 추가
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence
            else:
                # 사이즈를 넘으면 현재 청크 저장 후 초기화
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # overlap을 고려하여 이전 청크의 뒷부분을 가져올 수도 있지만
                # 여기서는 문장이 너무 길 경우 강제로 자르는 로직만 추가하거나
                # 그냥 새 청크로 시작함 (단순화)
                if len(sentence) > chunk_size:
                    # 문장 자체가 너무 길면 강제 분할
                    for i in range(0, len(sentence), chunk_size - overlap):
                        chunks.append(sentence[i:i + chunk_size])
                    current_chunk = ""  # 이미 다 처리했으므로
                else:
                    current_chunk = sentence

        # 마지막 남은 청크 처리
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def get_embeddings(self, text_list: List[str], batch_size: int = 100) -> np.array:
        """
        텍스트 리스트를 배치 단위로 나누어 임베딩을 생성합니다.
        OpenAI text-embedding-3-small 모델 사용
        """
        if not text_list:
            print("임베딩을 생성할 텍스트가 없습니다.")
            return np.array([], dtype='float32')

        all_embeddings = []
        total = len(text_list)
        print(f"총 {total}개의 텍스트 임베딩 생성 시작 (배치 크기: {batch_size})...")

        for i in range(0, total, batch_size):
            batch = text_list[i: i + batch_size]
            try:
                # OpenAI Embeddings API 호출
                response = self.client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch
                )

                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)

                # 진행 상황 출력
                print(f"  - {min(i + batch_size, total)}/{total} 완료")

                # API 속도 제한 방지를 위한 짧은 대기
                time.sleep(0.1)

            except Exception as e:
                print(f"  - 배치 {i}~{i + batch_size} 처리 중 에러: {e}")
                return None

        return np.array(all_embeddings, dtype='float32')

    def build_index(self, texts: List[str], titles: List[str]):
        """
        텍스트와 제목 리스트를 받아 임베딩을 생성하고 인덱스를 구축합니다.
        """
        if not texts:
            print("인덱싱할 텍스트가 없습니다.")
            return

        embeddings = self.get_embeddings(texts)
        if embeddings is None or len(embeddings) == 0:
            print("임베딩 생성 실패 또는 결과 없음")
            return

        d = embeddings.shape[1]

        # FAISS 인덱스 생성
        faiss.normalize_L2(embeddings)  # 코사인 유사도를 위해 정규화
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)

        # 메타데이터 저장
        self.metadata = [{"title": t, "content": c} for t, c in zip(titles, texts)]

        print(f"인덱스 구축 완료. 크기: {self.index.ntotal}")

    def save(self):
        """인덱스와 메타데이터를 파일로 저장합니다."""
        if self.index:
            faiss.write_index(self.index, self.index_file)
        if self.metadata:
            with open(self.metadata_file, "wb") as f:
                pickle.dump(self.metadata, f)
        print(f"저장 완료: {self.index_file}, {self.metadata_file}")

    def load(self):
        """파일에서 인덱스와 메타데이터를 불러옵니다."""
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"로드 완료. 인덱스 크기: {self.index.ntotal}")
            return True
        return False

    def search(self, query: str, k: int = 3):
        """쿼리에 대해 유사한 문서를 검색합니다."""
        if not self.index:
            print("인덱스가 비어있습니다.")
            return []

        # 쿼리 임베딩 (OpenAI API)
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[query]
            )
            query_vec = np.array([response.data[0].embedding], dtype='float32')

            faiss.normalize_L2(query_vec)

            distances, indices = self.index.search(query_vec, k)

            results = []
            for i in range(k):
                idx = indices[0][i]
                score = distances[0][i]
                if idx < len(self.metadata) and idx != -1:
                    item = self.metadata[idx]
                    results.append({
                        "score": score,
                        "title": item['title'],
                        "content": item['content']
                    })
            return results
        except Exception as e:
            print(f"검색 중 오류: {e}")
            return []
