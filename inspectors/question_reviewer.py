# GIA/inspectors/question_reviewer.py

from __future__ import annotations
from typing import Dict, Any, List
import re
from utils.llm_adapter import review_hypotheses_llm


# 필터링할 금지/필수 패턴 정의
FORBIDDEN_PATTERNS = [
    re.compile(r"(절차|방법|어떻게|무엇|어떤 종류|이유).*(가요|까요)\?$"),
    re.compile(r"설명하시오\.$"),
]
REQUIRED_PATTERNS = [
    re.compile(r"(개수|모두|전체|총|%|비율)"),
    re.compile(r"(=|==|!=|>|<|≥|≤)"),
    re.compile(r"(없음|존재하지 않음|설정되지 않음|누락|불일치)"),
]
def llm_reviewer(hypotheses: List[Dict[str, Any]], capabilities: Dict[str, Any], score_threshold: int = 10) -> List[Dict[str, Any]]:
    """
    LLM을 이용해 가설을 평가하고, 기준 점수 미달 항목을 제거합니다.
    """
    if not hypotheses:
        return []

    print(f"[Reviewer] LLM 리뷰어 호출: {len(hypotheses)}개 가설 평가 시작...")
    
    # LLM 어댑터를 통해 리뷰 요청
    reviews = review_hypotheses_llm(hypotheses, capabilities)
    
    if not reviews:
        print("[Reviewer] LLM 리뷰어가 유효한 응답을 반환하지 않았습니다. 모든 가설을 통과시킵니다.")
        return hypotheses

    recommended_indices = set()
    for review in reviews:
        if review.get("is_recommended") and review.get("total_score", 0) >= score_threshold:
            idx = review.get("hypothesis_index")
            if isinstance(idx, int):
                recommended_indices.add(idx)
        print(f"[Reviewer] 평가: idx={review.get('hypothesis_index')}, 점수={review.get('total_score')}, 추천={review.get('is_recommended')}, 근거={review.get('justification')}")

    reviewed_hypotheses = [
        hypo for i, hypo in enumerate(hypotheses) if i in recommended_indices
    ]
    
    print(f"[Reviewer] LLM 리뷰 완료: {len(reviewed_hypotheses)}개 가설 최종 선택 (기준 점수: {score_threshold}점 이상)")
    return reviewed_hypotheses


def heuristic_filter(hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    규칙 기반으로 품질이 낮은 가설을 필터링합니다.
    - 모호하거나 개방형 질문 제거
    - 정량적/조건적 표현이 없는 질문 제거
    """
    filtered = []
    for hypo in hypotheses:
        question = hypo.get("question", "").strip()
        if not question:
            continue

        # 1. 금지 패턴 검사: 모호한 질문 제거
        is_forbidden = False
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(question):
                print(f"[Reviewer] 금지된 패턴 포함으로 질문 필터링: {question}")
                is_forbidden = True
                break
        if is_forbidden:
            continue

        # 2. 필수 패턴 검사: 구체적인 조건이 없는 질문 제거
        has_required = False
        for pattern in REQUIRED_PATTERNS:
            if pattern.search(question):
                has_required = True
                break
        
        # 'intent_hint'에 조건이 명시된 경우도 통과
        intent_hint = hypo.get("intent_hint", {})
        if intent_hint.get("metric") and intent_hint.get("scope") is not None:
             # metric이 bool 타입으로 끝나는 경우, 질문 자체로 조건이 됨
            if intent_hint['metric'].endswith('_bool'):
                has_required = True
        
        if not has_required:
            print(f"[Reviewer] 필수 조건 미포함으로 질문 필터링: {question}")
            continue

        filtered.append(hypo)
        
    return filtered