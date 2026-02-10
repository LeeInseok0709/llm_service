import os
import json
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Awaitable, Dict, Any, List, Optional

from dotenv import load_dotenv

# LLM 라이브러리 임포트
from google import genai
from openai import OpenAI


# ----------------------------\
# 0) 환경 변수 / 클라이언트 초기화
# ----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GEMINI_API_KEY:
    print("[경고] GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
if not OPENAI_API_KEY:
    print("[경고] OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ----------------------------
# 1) 모델 호출 (비동기 래퍼)
# ----------------------------
async def get_gemini_response(prompt: str) -> str:
    if not gemini_client:
        return "[Gemini 오류] GEMINI_API_KEY가 설정되지 않았습니다."

    try:
        resp = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model="gemini-2.5-flash",
            contents=prompt
        )
        text = getattr(resp, "text", None)
        return text if text is not None else str(resp)
    except Exception as e:
        return f"[Gemini 오류] {type(e).__name__}: {e}"


async def get_openai_response(prompt: str) -> str:
    if not openai_client:
        return "[ChatGPT 오류] OPENAI_API_KEY가 설정되지 않았습니다."

    try:
        # Responses API (권장 흐름)
        resp = await asyncio.to_thread(
            openai_client.responses.create,
            model="gpt-4o-mini",
            input=prompt
        )
        return resp.output_text
    except Exception as e:
        msg = str(e)
        if "You exceeded your current quota" in msg:
            return (
                "[ChatGPT 오류] OpenAI API 쿼터(또는 요금제 한도)를 초과했습니다. "
                "플랜/결제/사용량을 확인해야 합니다.\n"
                f"원본 메시지: {msg}"
            )
        if "Rate limit" in msg or "Too many requests" in msg:
            return (
                "[ChatGPT 오류] 요청이 너무 빠르게 반복되거나 rate limit에 걸렸습니다.\n"
                f"원본 메시지: {msg}"
            )
        return f"[ChatGPT 일반 오류] {type(e).__name__}: {msg}"


async def run_comparison(prompt: str) -> Dict[str, str]:
    """두 모델 병렬 호출 → 결과 dict로 반환"""
    tasks = [
        get_gemini_response(prompt),
        get_openai_response(prompt),
    ]
    gem_text, oa_text = await asyncio.gather(*tasks)
    return {"Gemini": gem_text, "ChatGPT": oa_text}


# ----------------------------
# 2) 단계 확장(훅) 가능한 세션 설계
# ----------------------------
PostChoiceHook = Callable[[Dict[str, Any]], Awaitable[None]]
# ctx에는 question/responses/selected/winner_text/history 같은 걸 넣어둘 예정

@dataclass
class SessionConfig:
    show_full_text: bool = True
    save_history_jsonl: Optional[str] = None  # 예: "history.jsonl"


@dataclass
class CompareSession:
    config: SessionConfig = field(default_factory=SessionConfig)
    history: List[Dict[str, Any]] = field(default_factory=list)

    # ✅ "2 다음 단계"를 여기다가 마음대로 추가
    post_choice_hooks: List[PostChoiceHook] = field(default_factory=list)

    async def one_round(self, question: str) -> None:
        print(f"\n>> 사용자 질문: {question}\n")

        responses = await run_comparison(question)

        self._print_outputs(responses)

        selected = self._ask_choice()  # 1: Gemini / 2: ChatGPT
        winner_name = "Gemini" if selected == "1" else "ChatGPT"
        winner_text = responses[winner_name]

        # (2) 어떤 모델이 선택됐는지 알려줌
        print(f"\n✅ 선택된 모델: {winner_name}\n")

        # 라운드 기록
        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "question": question,
            "responses": responses,
            "selected": winner_name,
        }
        self.history.append(record)

        # 필요하면 파일로 누적 저장
        if self.config.save_history_jsonl:
            with open(self.config.save_history_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # (2) 다음에 추가될 “새 단계”는 여기서 훅으로 실행
        ctx = {
            "question": question,
            "responses": responses,
            "selected": winner_name,
            "winner_text": winner_text,
            "history": self.history,
        }
        for hook in self.post_choice_hooks:
            await hook(ctx)

    def _print_outputs(self, responses: Dict[str, str]) -> None:
        print("========== 모델 출력 ==========")
        for name in ["Gemini", "ChatGPT"]:
            text = responses.get(name, "")
            print(f"\n--- [ {name} ] ---")
            if self.config.show_full_text:
                print(text)
            else:
                preview = text.replace("\n", " ")
                print(preview[:400] + ("..." if len(preview) > 400 else ""))
        print("\n===============================")

    def _ask_choice(self) -> str:
        while True:
            choice = input("\n>> 어떤 결과를 선택하시겠어요? (1: Gemini / 2: ChatGPT / q: 종료): ").strip().lower()
            if choice in {"1", "2"}:
                return choice
            if choice in {"q", "quit", "exit"}:
                raise SystemExit
            print("입력이 올바르지 않습니다. 1 또는 2 (또는 q)를 입력해 주세요.")


# ----------------------------
# 3) 훅 예시 (원하시면 여기서 마음대로 확장)
# ----------------------------
async def hook_print_winner_first_200(ctx: Dict[str, Any]) -> None:
    """예시 훅: 선택된 답변 앞부분만 다시 보여주기"""
    winner = ctx["selected"]
    text = ctx["winner_text"].replace("\n", " ")
    print(f"🧩 (훅 예시) {winner} 답변 요약(200자): {text[:200]}{'...' if len(text) > 200 else ''}\n")


# ----------------------------
# 4) 메인 루프 (1~4 반복)
# ----------------------------
async def main():
    print("--- LLM 비교 서비스 (반복 모드) ---")

    session = CompareSession(
        config=SessionConfig(
            show_full_text=True,
            save_history_jsonl=None,  # 필요하면 "history.jsonl" 넣으세요
        ),
        post_choice_hooks=[
            # ✅ 여기에 '2 다음 단계'를 계속 추가하면 됨
            # hook_print_winner_first_200,
        ],
    )

    while True:
        q = input("\n>> 다음 질문을 입력하세요 (q: 종료): ").strip()
        if not q or q.lower() in {"q", "quit", "exit"}:
            print("프로그램을 종료합니다.")
            break

        try:
            await session.one_round(q)
        except SystemExit:
            print("프로그램을 종료합니다.")
            break


if __name__ == "__main__":
    asyncio.run(main())
