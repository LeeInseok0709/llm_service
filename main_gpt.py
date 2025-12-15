import os
from dotenv import load_dotenv
import asyncio

# LLM 라이브러리 임포트
from google import genai
from openai import OpenAI

# 1. 환경 변수 로드
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 2. 클라이언트 초기화
if not GEMINI_API_KEY:
    print("[경고] GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
if not OPENAI_API_KEY:
    print("[경고] OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- 모델별 비동기 함수 정의 ---

async def get_gemini_response(prompt: str):
    """Gemini 모델에게 질문하고 응답을 받는 비동기 함수"""
    try:
        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model="gemini-2.5-flash",
            contents=prompt
        )
        # 일부 Gemini 클라이언트는 .text 대신 candidates 구조를 가지기도 하므로 방어 코드
        text = getattr(response, "text", None)
        if text is None:
            text = str(response)
        return "Gemini", text
    except Exception as e:
        return "Gemini", f"[Gemini 오류] {type(e).__name__}: {e}"

async def get_openai_response(prompt: str):
    """OpenAI 모델에게 질문하고 응답을 받는 비동기 함수"""
    if not OPENAI_API_KEY:
        return "ChatGPT", "[ChatGPT 오류] OPENAI_API_KEY가 설정되지 않았습니다."

    try:
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-4o-mini",  # gpt-4o-mini, gpt-3.5-turbo 등
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.choices[0].message.content
        return "ChatGPT", text

    except Exception as e:
        # OpenAI 429/쿼터 초과 구분을 위해 메시지 내용 확인
        msg = str(e)
        if "You exceeded your current quota" in msg:
            return (
                "ChatGPT",
                "[ChatGPT 오류] OpenAI API 쿼터(또는 요금제 한도)를 초과했습니다. "
                "플랜/결제/사용량을 확인해야 합니다.\n"
                f"원본 메시지: {msg}"
            )
        elif "Rate limit" in msg or "Too many requests" in msg:
            return (
                "ChatGPT",
                "[ChatGPT 오류] 요청이 너무 빠르게 반복되거나 rate limit에 걸렸습니다.\n"
                f"원본 메시지: {msg}"
            )
        else:
            return "ChatGPT", f"[ChatGPT 일반 오류] {type(e).__name__}: {msg}"

# --- 병렬 실행 및 결과 반환 함수 ---

async def run_comparison(user_prompt: str):
    """두 모델에게 동시에 질문하고 응답을 받습니다."""
    print(f"\n>> 사용자 질문: {user_prompt}\n")

    tasks = [
        get_gemini_response(user_prompt),
        get_openai_response(user_prompt)
    ]

    results = await asyncio.gather(*tasks)

    responses = {}
    for model_name, response_text in results:
        responses[model_name] = response_text
        print(f"--- [ {model_name} 응답 (부분) ] ---")
        # 응답이 에러 메시지일 수도 있으니 그냥 150자만 미리보기
        preview = response_text.replace("\n", " ")
        print(preview[:150] + "...")
        print("-" * 25)

    return responses

# --- 메인 실행 ---

if __name__ == "__main__":
    print("--- LLM 비교 서비스 실행 ---")
    
    # question을 사용자 입력으로 받도록 수정
    question = input(">> 비교할 AI 질문을 입력하세요: ")

    if question.strip():
        results = asyncio.run(run_comparison(question))
        # 이후 비교/선택 로직 추가 가능
        # print(results)
    else:
        print("질문이 입력되지 않아 프로그램을 종료합니다.")