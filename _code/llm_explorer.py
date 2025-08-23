from __future__ import annotations
from typing import Dict, Any, List

from utils.llm_adapter import generate_hypothesis_llm, parse_intent_llm
from utils.builder_core import list_available_metrics, make_grounding
from inspectors.question_reviewer import heuristic_filter, llm_reviewer

def _lint_intent(intent: Dict[str, Any], metrics: List[str]) -> Dict[str, Any]:
    it = dict(intent or {})
    q = (it.get("_q") or "").lower()

    # metric 유효성 보정 및 자동 매핑
    m = it.get("metric")
    if m not in metrics:
        if "ssh" in q and ("모든" in q or "전체" in q):
            it["metric"] = "ssh_all_enabled_bool" if "ssh_all_enabled_bool" in metrics else "ssh_missing_count"
        elif "bgp" in q and any(k in q for k in ["영향", "오류", "일관", "피어링"]):
            pref = [
                "ibgp_missing_pairs_count",
                "ibgp_under_peered_count",
                "ibgp_missing_pairs",
            ]
            it["metric"] = next((x for x in pref if x in metrics), it.get("metric"))
        elif "l3vpn" in q or "개통" in q or "절차" in q:
            pref = ["vrf_rt_list_per_device", "vrf_rd_map", "vrf_bind_map", "l2vpn_pairs"]
            it["metric"] = next((x for x in pref if x in metrics), it.get("metric"))

    # scope 키 정리(alias)
    sc = dict(it.get("scope") or {})
    if "interface" in sc and "if" not in sc:
        sc["if"] = sc.get("interface")
    it["scope"] = sc

    return it


def _coerce_hypothesis_list(raw: Any) -> List[Dict[str, Any]]:
    """
    LLM 출력(raw)을 가능한 한 가설 리스트로 보정한다.
    - 허용 형태: list[dict], dict(내부에 list 포함), 단일 dict, list[str]
    - 필수 최소 조건: question 문자열 1개
    - 누락 필드(intent_hint, expected_condition 등)는 기본값 자동 보정
    """
    # 1) 1차 원본 구조 로깅 (디버깅 편의)
    try:
        print(f"[_coerce] raw_type={type(raw).__name__} keys={list(raw.keys())[:6] if isinstance(raw, dict) else 'NA'} len={len(raw) if hasattr(raw,'__len__') else 'NA'}")
    except Exception:
        pass

    items: List[Any] = []
    # a) 이미 리스트
    if isinstance(raw, list):
        items = raw
    # b) dict → 값 중 첫 list
    elif isinstance(raw, dict):
        # 단일 가설 객체일 가능성
        if isinstance(raw.get("question"), str):
            items = [raw]
        else:
            for v in raw.values():
                if isinstance(v, list):
                    items = v
                    break
            if not items:
                # wrapping 형태 (예: {"data": {"items": [...]}} ) 탐색 2단계
                for v in raw.values():
                    if isinstance(v, dict):
                        for vv in v.values():
                            if isinstance(vv, list):
                                items = vv; break
                        if items:
                            break
        if not items:
            items = [raw]
    else:
        # 문자열 덩어리일 수 있음 → JSON 일부 누락된 경우 포기
        if isinstance(raw, str) and raw.strip():
            print("[_coerce] raw is plain string; no JSON array parsed")
        return []

    # 2) 항목 정규화
    out: List[Dict[str, Any]] = []
    for idx, it in enumerate(items):
        if isinstance(it, str):
            q = it.strip()
            if not q:
                continue
            out.append({
                "question": q,
                "hypothesis_type": "ImpactAnalysis",
                "intent_hint": {"metric": "", "scope": {}},
                "expected_condition": "",
                "reasoning_steps": "(LLM raw string → 보정)",
                "cited_values": {}
            })
            continue
        if not isinstance(it, dict):
            continue
        q = (it.get("question") or "").strip()
        if not q:
            continue
        # 누락 필드 보정
        hyp = dict(it)
        hyp.setdefault("hypothesis_type", hyp.get("type") or "ImpactAnalysis")
        hyp.setdefault("intent_hint", {
            "metric": (hyp.get("intent_hint") or {}).get("metric") if isinstance(hyp.get("intent_hint"), dict) else "",
            "scope": (hyp.get("intent_hint") or {}).get("scope") if isinstance(hyp.get("intent_hint"), dict) else {}
        })
        if not isinstance(hyp.get("intent_hint"), dict):
            hyp["intent_hint"] = {"metric": "", "scope": {}}
        ih = hyp["intent_hint"]
        ih.setdefault("metric", "")
        ih.setdefault("scope", {})
        hyp.setdefault("expected_condition", it.get("predicted_answer") or "")
        hyp.setdefault("reasoning_steps", it.get("rationale") or it.get("reasoning") or "")
        hyp.setdefault("cited_values", it.get("cited_values") or {})
        # 최소 기준: question
        out.append(hyp)

    print(f"[_coerce] accepted {len(out)} / raw_items {len(items)}")
    return out


class LLMExplorer:
    def from_llm(self, facts: Dict[str, Any], policies: Any, n_hypotheses: int = 5) -> List[Dict[str, Any]]:
        translated: List[Dict[str, Any]] = []
        metrics: List[str] = list_available_metrics()
        focused_ctx = make_grounding(facts) if isinstance(facts, (dict, list)) else facts
        raw = generate_hypothesis_llm(focused_ctx, policies, n_hypotheses=n_hypotheses, builder_metrics=metrics)
        hypos_raw = _coerce_hypothesis_list(raw)
        if not hypos_raw:
            print("[LLMExplorer] (fallback) No hypotheses parsed → 최소 question-only salvage 시도 건너뜀")
        hypos_heuristic = heuristic_filter(hypos_raw) if hypos_raw else []
        print(f"[LLMExplorer] 원본 가설 {len(hypos_raw)}개 -> 필터 후 {len(hypos_heuristic)}개")
        hypos = llm_reviewer(hypos_heuristic, focused_ctx) if hypos_heuristic else []
        # if not hypos:
        #     print("[LLMExplorer] LLM 리뷰어가 유효한 응답을 반환하지 않았습니다. 모든 가설을 통과시킵니다.")
        #     return hypos_raw
        # print(f"[LLMExplorer] LLM 리뷰 완료: {len(hypos)}개 가설 최종 선택")


        for h in hypos:
            question = h.get("question")
            if not question:
                continue
            try:
                hint_metric = (h.get("intent_hint") or {}).get("metric") or h.get("suggested_metric")
                hint_scope = (h.get("intent_hint") or {}).get("scope") or h.get("suggested_scope")
                cited_values = h.get("cited_values", {})
                intent = parse_intent_llm(
                    question,
                    metrics,
                    hint_metric=hint_metric,
                    hint_scope=hint_scope,
                    cited_values=cited_values
                )
                if not isinstance(intent, dict):
                    intent = {}
                if not intent.get("metric"):
                    ql = (question or "").lower(); fallback_metric = hint_metric
                    if not fallback_metric:
                        if "ssh" in ql and ("모든" in ql or "전체" in ql):
                            fallback_metric = "ssh_all_enabled_bool" if "ssh_all_enabled_bool" in metrics else "ssh_missing_count"
                        elif any(k in ql for k in ["bgp","피어","일관"]):
                            for cand in ["ibgp_missing_pairs_count","ibgp_under_peered_count","ibgp_missing_pairs"]:
                                if cand in metrics: fallback_metric = cand; break
                        elif "l2vpn" in ql:
                            for cand in ["l2vpn_mismatch_count","l2vpn_unidirectional_pairs"]:
                                if cand in metrics: fallback_metric = cand; break
                        elif any(k in ql for k in ["vrf","l3vpn","route-target","rt"]):
                            for cand in ["vrf_without_rt_pairs","vrf_rt_list_per_device","vrf_rd_map","vrf_bind_map"]:
                                if cand in metrics: fallback_metric = cand; break
                        else:
                            fallback_metric = "ssh_missing_count" if "ssh_missing_count" in metrics else metrics[0]
                    intent["metric"] = fallback_metric
                if not isinstance(intent.get("scope"), dict):
                    intent["scope"] = {}
                intent["_q"] = question
                intent = _lint_intent(intent, metrics)
                translated.append({
                    "origin": "LLM_Explorer",
                    "hypothesis": h,
                    "intent": intent,
                })
            except Exception as e:
                print(f"[LLMExplorer] intent 변환 실패: {e}")
                continue
        return translated
