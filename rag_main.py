import os
from full_pipeline import get_video_transcript, refine_script
from vector import HKTVectorStore, CSV_PATH

def main():
    print("=== YouTube RAG Pipeline Start (Class-based) ===\n")
    
    # 1. Vector Store 초기화
    store = HKTVectorStore() 
    
    # 2. 로드 또는 새로 구축
    if store.load():
        print(f">> 기존 라이브러리 로드 완료. (문서 수: {len(store.metadata)})")
    else:
        print(">> 기존 라이브러리가 없어 새로 구축합니다.")
        df = store.load_data(CSV_PATH)
        if df is None:
            print("Error: CSV 파일을 찾을 수 없습니다.")
            return

        all_texts = []
        all_titles = []
        
        print(">> 데이터 청킹(Chunking) 및 준비 중...")
        for _, row in df.iterrows():
            content = str(row.get('content', ''))
            title = row.get('title', 'No Title')
            
            # 스마트 청킹 사용
            chunks = store.chunk_text(content)
            
            for c in chunks:
                all_texts.append(c)
                all_titles.append(title)
        
        print(f">> {len(all_texts)}개의 청크 생성됨. 인덱싱 시작...")
        store.build_index(all_texts, all_titles)
        store.save()
        print(">> 인덱싱 구축 및 저장 완료.\n")

    # 3. YouTube Video 처리
    VIDEO_ID = "TCaDxE3wXsI" 
    print(f">> YouTube 스크립트 처리 중 ({VIDEO_ID})...")
    
    raw_script = get_video_transcript(VIDEO_ID)
    if not raw_script: return

    refined_script = refine_script(raw_script)
    final_query_text = refined_script if refined_script else raw_script

    # 4. 검색 수행 (검색 쿼리가 너무 길면 앞부분만 사용하거나 요약해서 사용)
    # RAG에서는 보통 질문(Query)을 던지지만, 여기서는 스크립트 내용과 유사한 문서를 찾는 것이므로
    # 스크립트 전체를 쿼리로 쓰기보다, 핵심 내용을 추출하거나 앞부분을 사용합니다.
    query_text = final_query_text 
    
    print(f"\n>> 검색 수행 중 (Query 길이: {len(query_text)}자)...")
    results = store.search(query_text, k=4)

    print("\n" + "="*50)
    print(f"   RAG 검색 결과 (Top {len(results)})")
    print("="*50)
    
    for i, res in enumerate(results):
        print(f"\n[{i+1}] [유사도: {res['score']:.4f}] {res['title']}")
        print(f"   내용: {res['content']}") 
        
    if not results:
        print("검색 결과가 없습니다.")

if __name__ == "__main__":
    main()


'''
1	보수논객
2	좌파논객
3	보수렉카
4	중립
5	좌파렉카
'''