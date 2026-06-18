from __future__ import annotations
import json
import os
import re
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM

FAILURE_MODE_BY_QID = {}
_USE_MOCK = not os.environ.get("OPENAI_API_KEY")

_client = None
last_tokens = {"prompt": 0, "completion": 0, "total": 0}

# ── Mock fallback ──────────────────────────────────────────────
_FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}

def _mock_actor(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if example.qid not in _FIRST_ATTEMPT_WRONG:
        return example.gold_answer
    if agent_type == "react":
        return _FIRST_ATTEMPT_WRONG[example.qid]
    if attempt_id == 1 and not reflection_memory:
        return _FIRST_ATTEMPT_WRONG[example.qid]
    return example.gold_answer

def _mock_evaluator(example: QAExample, answer: str) -> JudgeResult:
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
    return JudgeResult(score=0, reason="The final answer is incorrect.", missing_evidence=["Need to ground the answer in the context."], spurious_claims=[answer])

def _mock_reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    return ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial answer is not enough; the final answer must complete all hops.", next_strategy="Verify the final entity against the second paragraph before answering.")

# ── OpenAI runtime ─────────────────────────────────────────────
def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI()
    return _client

def _get_model() -> str:
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

def _chat(system: str, user: str) -> str:
    global last_tokens
    client = _get_client()
    resp = client.chat.completions.create(
        model=_get_model(),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    usage = resp.usage
    if usage:
        last_tokens = {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens, "total": usage.total_tokens}
    return resp.choices[0].message.content.strip()

def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    return {}

# ── Public API (auto-select mock or openai) ────────────────────
def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if _USE_MOCK:
        answer = _mock_actor(example, attempt_id, agent_type, reflection_memory)
        last_tokens["total"] = len(answer.split()) + len(example.question.split()) + sum(len(c.text.split()) for c in example.context)
        return answer
    context_str = "\n\n".join(f"[{c.title}] {c.text}" for c in example.context)
    reflection_str = ""
    if reflection_memory:
        reflection_str = "\n\nLessons from previous failed attempts:\n" + "\n".join(f"- {m}" for m in reflection_memory)
    user_msg = f"Context:\n{context_str}{reflection_str}\n\nQuestion: {example.question}\n\nAnswer:"
    return _chat(ACTOR_SYSTEM, user_msg)

def evaluator(example: QAExample, answer: str) -> JudgeResult:
    if _USE_MOCK:
        return _mock_evaluator(example, answer)
    user_msg = (
        f"question: {example.question}\n"
        f"gold_answer: {example.gold_answer}\n"
        f"predicted_answer: {answer}\n\n"
        f"Return ONLY a JSON object."
    )
    raw = _chat(EVALUATOR_SYSTEM, user_msg)
    data = _parse_json(raw)
    return JudgeResult(
        score=data.get("score", 0),
        reason=data.get("reason", ""),
        missing_evidence=data.get("missing_evidence", []),
        spurious_claims=data.get("spurious_claims", []),
    )

def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    if _USE_MOCK:
        return _mock_reflector(example, attempt_id, judge)
    user_msg = (
        f"question: {example.question}\n"
        f"attempt_answer: (see below)\n"
        f"failure_reason: {judge.reason}\n"
        f"gold_answer: {example.gold_answer}\n\n"
        f"Return ONLY a JSON object."
    )
    raw = _chat(REFLECTOR_SYSTEM, user_msg)
    data = _parse_json(raw)
    return ReflectionEntry(
        attempt_id=data.get("attempt_id", attempt_id),
        failure_reason=data.get("failure_reason", judge.reason),
        lesson=data.get("lesson", ""),
        next_strategy=data.get("next_strategy", ""),
    )
