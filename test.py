# The client gets the API key from the environment variable `GEMINI_API_KEY`.
from google import genai
import youtubescript

client = genai.Client(api_key="AIzaSyAa-TXqxVWLX3ITxz0FJf_3tAfEKuoibck")

### 필수 아님

# Count tokens using the new client method.
# total_tokens = client.models.count_tokens(
#     model="gemini-2.5-flash",
#     contents=prompt
# )
# print("total_tokens: ", total_tokens)
# # ( e.g., total_tokens: 10 )

SYSTEM_PROMPT = """
너는 뉴스 자막 편집 전문가다.

- 원문 의미를 절대 바꾸지 마라
- 요약하지 말고 전체 내용을 유지하라
- 자동 생성 자막의 오탈자, 중복, 잘린 문장을 복원하라
- 줄바꿈은 문단 단위로 정리하라
- 화자 기호는 제거하라
- 추측이나 해석을 추가하지 마라
"""

def clean_script(raw_script: str) -> str:
    prompt = f"""
SYSTEM:
{SYSTEM_PROMPT}

INPUT:
{raw_script}

OUTPUT:
정제된 한국어 뉴스 문장
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    print(response.usage_metadata)
    return response.text
