import os
from dotenv import load_dotenv
import asyncio

# LLM 라이브러리 임포트
from google import genai
from openai import OpenAI

# 1. 환경 변수 로드
load_dotenv()
# Conda 환경에서는 API 키를 직접 코드에 두지 않고 환경 변수에서 가져옵니다.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 2. 클라이언트 초기화
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- 모델별 비동기 함수 정의 ---

async def get_gemini_response(prompt):
    """Gemini 모델에게 질문하고 응답을 받는 비동기 함수"""
    try:
        # 동기 호출 함수를 비동기로 실행
        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model='gemini-2.5-flash',
            contents=prompt
        )
        return "Gemini", response.text
    except Exception as e:
        return "Gemini", f"오류 발생: {e}"

async def get_openai_response(prompt):
    """OpenAI 모델에게 질문하고 응답을 받는 비동기 함수"""
    try:
        # 동기 호출 함수를 비동기로 실행
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-4o-mini", # gpt-4o-mini   gpt-3.5-turbo
            messages=[{"role": "user", "content": prompt}]
        )
        return "ChatGPT", response.choices[0].message.content
    except Exception as e:
        return "ChatGPT", f"오류 발생: {e}"

# --- 병렬 실행 및 결과 반환 함수 ---

async def run_comparison(user_prompt):
    """두 모델에게 동시에 질문하고 응답을 받습니다."""
    print(f"\n>> 사용자 질문: {user_prompt}\n")

    # 두 모델의 함수를 동시에 실행 (병렬 처리)
    tasks = [
        get_gemini_response(user_prompt),
        get_openai_response(user_prompt)
    ]

    # 모든 작업이 완료될 때까지 기다림
    results = await asyncio.gather(*tasks)

    # 응답 출력 및 정리
    responses = {}
    for model_name, response_text in results:
        responses[model_name] = response_text
        print(f"--- [ {model_name} 응답 (부분) ] ---")
        print(response_text[:150] + "...")
        print("-" * 25)

    return responses

# --- 메인 실행 ---
if __name__ == "__main__":
    question = "AI 기술이 미래 교육에 미칠 긍정적 영향과 부정적 영향을 3가지씩 설명해줘."
    print("--- LLM 비교 서비스 실행 ---")
    
    # 비동기 함수 실행
    results = asyncio.run(run_comparison(question))
    
    # 4단계: 응답 비교 및 선택 로직이 여기에 추가됩니다.
    # print(results)