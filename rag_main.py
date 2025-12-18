import os
from full_pipeline import get_video_transcript, refine_script, client, LLM_MODEL
from vector import HKTVectorStore, CSV_PATH
import re
# rag_main.py



# OpenAI 클라이언트는 full_pipeline에서 가져옴

def rerank_candidates(query_chunk, candidates):
    """
    벡터 검색으로 나온 후보군(candidates) 중에서
    정치적 성향(Stance)과 논조(Tone)가 가장 일치하는 1개를 GPT-4o-mini가 선택합니다.
    """

    # 후보군 텍스트 포맷팅
    candidates_text = ""
    for i, cand in enumerate(candidates):
        candidates_text += f"[{i + 1}] {cand['content']}\n"

    system_prompt = """너는 정치적 성향 분석 전문가다.
주어진 [입력 문장]과 [후보 문장들]을 비교하여, **'정치적 성향(Political Stance)'과 '비판 대상'이 가장 일치하는 문장** 하나를 골라라.

INSTRUCTIONS:
1. 단순 단어 매칭이 아니라, **문장의 의도와 편향성**이 일치해야 한다.
2. 예를 들어, 입력이 "우파 비판"이면 후보도 "우파 비판"이어야 한다. 입력이 "좌파 비판"인데 후보가 "우파 비판"이면 절대 선택하면 안 된다.
3. [입력 문장]이 단순 욕설이거나, [후보 문장들] 중에 논리적으로 유사한 게 전혀 없다면 "NONE"을 반환하라.
4. 가장 적절한 후보가 있다면 그 번호(예: 1, 2, 3...)만 딱 출력하라. 사족 달지 마라."""

    user_prompt = f"""[입력 문장]:
{query_chunk}

[후보 문장들]:
{candidates_text}

OUTPUT (번호 or NONE):"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=10
        )
        result = response.choices[0].message.content.strip()

        if "NONE" in result:
            return None

        # 숫자만 추출
        match = re.search(r'\d+', result)
        if match:
            idx = int(match.group()) - 1  # 0-indexed로 변환
            if 0 <= idx < len(candidates):
                return candidates[idx]
        return None
    except Exception as e:
        print(f"Reranking Error: {e}")
        return None


# ============================================================
# 편향도 계산 함수
# ============================================================
def calculate_bias_score(matches):
    """
    Top N 매칭 결과를 받아 가중평균 편향도를 계산합니다.

    레이블 매핑:
        좌파렉카: -2, 진보논객/좌파논객: -1, 중도/중립: 0, 보수논객: +1, 보수렉카: +2

    Returns: {"score": float, "label": str, "detail": dict}
    """
    label_to_score = {
        "좌파렉카": -2,
        "좌파논객": -1,
        "진보논객": -1,
        "중도": 0,
        "중립": 0,
        "보수논객": 1,
        "보수렉카": 2
    }

    # CSV title (숫자)를 label로 변환하는 매핑
    title_to_label = {
        1: "보수논객",
        2: "좌파논객",
        3: "보수렉카",
        4: "중립",
        5: "좌파렉카"
    }

    if not matches:
        return {"score": 0.0, "label": "분석 불가", "detail": {}}

    weighted_sum = 0.0
    total_weight = 0.0
    label_counts = {}

    for m in matches:
        raw_title = m.get("found_title", "중립")
        similarity = m.get("score", 0.5)

        # title -> label 변환
        try:
            label = title_to_label.get(int(raw_title), str(raw_title))
        except:
            label = str(raw_title)

        # label counts
        label_counts[label] = label_counts.get(label, 0) + 1

        # weighted sum
        score = label_to_score.get(label, 0)
        weighted_sum += score * similarity
        total_weight += similarity

    # 최종 편향도
    if total_weight == 0:
        bias_score = 0.0
    else:
        bias_score = weighted_sum / total_weight

    # 해석
    if bias_score <= -1.5:
        bias_label = "극좌"
    elif bias_score <= -0.5:
        bias_label = "좌파"
    elif bias_score <= 0.5:
        bias_label = "중도"
    elif bias_score <= 1.5:
        bias_label = "보수"
    else:
        bias_label = "극우"

    return {
        "score": round(bias_score, 4) * 10,
        "label": bias_label,
        "detail": label_counts
    }


# ============================================================
# Service Layer: analyze_video (Controller에서 호출)
# ============================================================
def analyze_video(url: str, tag: str) -> dict:
    """
    YouTube 영상을 분석하여 다음을 반환합니다:
    {
        "summary": 뉴스 요약 (3줄),
        "biased_sentences": 편향이라 판단한 상위 2개 문장,
        "weight": 편향도 점수,
    }
    """
    # 1. Vector Store 초기화 및 로드
    store = HKTVectorStore()
    if not store.load():
        # 인덱스가 없으면 에러
        return {"error": "Vector Store not found. Build index first."}

    raw_script = "내란특검은 김건희 씨의 비상개엄 선포 관여 의혹은 사실이 아니라고 결론 내렸다고 밝혔습니다. >> 박지영 특별 검사보는 오늘 최종 수사 결과를 발표하며 명태균 사건 등에서 김씨의 개입이 나오긴 하지만 개엄 선포에 관여하거나 윤석열 전 대통령이 개엄을 선포한 이유는 아닌 것 같다고 말했습니다."
    # 2. YouTube 트랜스크립트 가져오기
    # raw_script = get_video_transcript(url)
    if not raw_script:
        return {"error": "Failed to fetch transcript."}

    # 3. 정제 + 요약
    print("refine")
    result = refine_script(raw_script)
    print("refine success")
    refined_text = result.get("refined", raw_script)
    summary = result.get("summary", "")
    print("refine text, summary")

    # 4. Chunking
    input_chunks = store.chunk_text(refined_text, chunk_size=100, overlap=10)
    print("chunking")

    # 5. RAG + Reranking
    all_matches = []
    for i, chunk in enumerate(input_chunks):
        if len(chunk) < 10:
            continue

        results = store.search(chunk, k=4)
        if not results:
            continue

        best = rerank_candidates(chunk, results)
        if best:
            all_matches.append({
                "index": i + 1,
                "input_chunk": chunk,
                "found_content": best['content'],
                "found_title": best['title'],
                "score": float(best['score'])  # float32 -> float 변환
            })

    # 6. 편향도 계산
    bias_result = calculate_bias_score(all_matches)

    # 7. 상위 2개 편향 문장 추출 (유사도 높은 순)
    all_matches.sort(key=lambda x: x['score'], reverse=True)
    top2 = all_matches[:2]
    biased_sentences = []
    for m in top2:
        biased_sentences.append(m["input_chunk"])

    return {
        "report_text": summary,
        "words": biased_sentences,
        "weight": bias_result["score"]
    }


def service(url: str, tag: str):
    """
    CLI 테스트용 main 함수.
    analyze_video() 서비스 함수를 호출하고 결과를 출력합니다.
    """
    import json

    print("=== YouTube RAG Pipeline (Service Test) ===\n")

    # 테스트할 비디오 ID
    # VIDEO_ID = "TCaDxE3wXsI"  # YTN
    # VIDEO_ID = "YxmUIfr6HmU"  # 극우 테스트용

    # print(f">> 분석 대상: {VIDEO_ID}\n")

    # 서비스 함수 호출
    result = analyze_video(url, tag)

    # 결과 출력
    print("\n" + "=" * 80)
    print("   [분석 결과 (Controller Response)]")
    print("=" * 80)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("=" * 80)

    return result

#
# if __name__ == "__main__":
#     main("https://www.youtube.com/watch?v=TCaDxE3wXsI", "YTN")