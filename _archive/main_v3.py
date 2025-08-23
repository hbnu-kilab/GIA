from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import argparse, json, time

from parsers.universal_parser import UniversalParser
from generators.rule_based_generator import RuleBasedGenerator, RuleBasedGeneratorConfig
from assemblers.test_assembler import TestAssembler, AssembleOptions
from inspectors.intent_inspector import IntentInspector


def _capabilities_from_facts(facts: Dict[str,Any]) -> Dict[str, Any]:
	devices = facts.get("devices", []) if isinstance(facts, dict) else (facts or [])
	hosts = sorted([(d.get("system",{}).get("hostname") or d.get("file")) for d in devices])
	vrfs=set(); by_as={}
	for d in devices:
		las = d.get("routing",{}).get("bgp",{}).get("local_as")
		if las: by_as.setdefault(las,0); by_as[las]+=1
		for v in d.get("routing",{}).get("bgp",{}).get("vrfs",[]) or []:
			if v.get("name"): vrfs.add(v["name"])
		for sv in d.get("services",{}).get("vrf",[]) or []:
			if sv.get("name"): vrfs.add(sv.get("name"))
	as_groups = [{"asn": k, "count": v} for k,v in by_as.items()]
	return {"hosts": hosts, "vrfs": sorted(list(vrfs)), "bgp_as_groups": as_groups}


def _scope_signature(t: dict) -> str:
	sc = (t.get("evidence_hint", {}) or {}).get("scope") or (t.get("intent") or {}).get("scope") or {}
	st = (sc.get("type") or "GLOBAL").upper()
	if st == "AS":
		return f"AS={sc.get('asn')}"
	if st == "DEVICE":
		return f"DEVICE={sc.get('host')}"
	if st == "VRF":
		return f"VRF={sc.get('vrf')}"
	if st == "DEVICE_VRF":
		return f"DEVICE_VRF={sc.get('host')}/{sc.get('vrf')}"
	if st == "DEVICE_IF":
		return f"DEVICE_IF={sc.get('host')}/{sc.get('if')}"
	return "GLOBAL"


def _write_html_report(out_dir: Path, inspected: Dict[str, List[Dict[str, Any]]], title: str) -> None:
	rows = []
	idx = 0
	for cat, arr in inspected.items():
		for t in arr:
			idx += 1
			q = (t.get("question") or "")
			metric = ((t.get("evidence_hint") or {}).get("metric")
					  or ((t.get("intent") or {}).get("metric")) or "")
			sig = f"{metric}/{_scope_signature(t)}" if metric else _scope_signature(t)
			expected = t.get("expected_answer",{}).get("value")
			expected_str = json.dumps(expected, ensure_ascii=False) if expected is not None else ""
			rows.append({
				"#": idx,
				"category": cat,
				"test_id": t.get("test_id"),
				"status": (t.get("verification") or {}).get("status") or "PASS",
				"needs_review": (t.get("verification") or {}).get("needs_review") if (t.get("verification") is not None) else False,
				"level": t.get("level") if (t.get("level") is not None) else (t.get("intent") or {}).get("level"),
				"origin": t.get("origin") or (t.get("intent") or {}).get("origin") or "Universal",
				"sig": sig,
				"question": q.replace("<", "&lt;")[:300],
				"expected": expected_str[:200]
			})
	total = len(rows)
	html = [
		"<!doctype html><meta charset='utf-8'><title>TestSet Report</title>",
		"<style>body{font-family:system-ui,Arial}table{border-collapse:collapse;width:100%}"
		"td,th{border:1px solid #ddd;padding:6px}th{background:#f8f9fb}"
		"tr:nth-child(even){background:#fafafa}"
		"h1 small{font-size:60%;color:#666}"
		".PASS{color:#137333}.FAIL{color:#b00020;font-weight:600}"
		".filters{margin:10px 0;display:flex;gap:10px;align-items:center;flex-wrap:wrap}"
		"select,input{padding:6px;border:1px solid #bbb;border-radius:6px}"
		".summary{margin:8px 0;color:#333}"
		"</style>",
		f"<h1>{title} <small>generated {time.strftime('%Y-%m-%d %H:%M:%S')}</small></h1>",
		f"<div class='summary'>총 {total}건</div>",
		"<div class='filters'>"
		"<label>Level <select id='fLevel'><option>All</option><option>1</option><option>2</option><option>3</option></select></label>"
		"<label>Origin <select id='fOrg'><option>All</option><option>Architect</option><option>Universal</option></select></label>"
		"<label>Category <select id='fCat'><option>All</option></select></label>"
		"<input id='fText' placeholder='검색(질문/시그널/정답)' size='30'/>"
		"</div>",
		"<table id='tbl'><thead><tr><th>#</th><th>Category</th><th>Status</th><th>Needs Review</th>"
		"<th>Level</th><th>Origin</th><th>Sig(metric/scope)</th><th>Test ID</th><th>Expected</th><th>Question</th></tr></thead><tbody>",
	]
	for r in rows:
		html.append(
			f"<tr data-level='{r['level']}' data-org='{r['origin']}' data-cat='{r['category']}'>"
			f"<td>{r['#']}</td>"
			f"<td>{r['category']}</td>"
			f"<td class='{r['status']}'>{r['status']}</td>"
			f"<td>{r['needs_review']}</td>"
			f"<td>{r['level']}</td>"
			f"<td>{r['origin']}</td>"
			f"<td>{r['sig']}</td>"
			f"<td>{r['test_id']}</td>"
			f"<td style='max-width:320px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis' title='{r['expected']}'>{r['expected']}</td>"
			f"<td style='max-width:520px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis' title='{r['question']}'>{r['question']}</td></tr>"
		)
	script = (
		"<script>const F=(id)=>document.getElementById(id);"
		"const cats=[...new Set([...document.querySelectorAll('#tbl tbody tr')].map(tr=>tr.dataset.cat))];"
		"cats.sort().forEach(c=>{const o=document.createElement('option');o.textContent=c;F('fCat').appendChild(o);});"
		"const flt=()=>{let L=F('fLevel').value,O=F('fOrg').value,C=F('fCat').value,T=F('fText').value.toLowerCase();"
		"let n=0;document.querySelectorAll('#tbl tbody tr').forEach(tr=>{let ok=true;"
		"if(L!='All' && tr.dataset.level!=L) ok=false;"
		"if(O!='All' && tr.dataset.org!=O) ok=false;"
		"if(C!='All' && tr.dataset.cat!=C) ok=false;"
		"const cells=[...tr.children].map(td=>td.textContent.toLowerCase()).join(' ');"
		"if(T && !cells.includes(T)) ok=false;"
		"tr.style.display=ok?'':'none'; if(ok) n++;});"
		"document.querySelector('.summary').textContent='총 '+n+'건';};"
		"['fLevel','fOrg','fCat','fText'].forEach(id=>F(id).addEventListener('input',flt));"
		"</script>"
	)
	html.append("</tbody></table>"+script)
	(out_dir / "report.html").write_text("\n".join(html), encoding="utf-8")


def _volume_boost(inspected: Dict[str, List[Dict[str, Any]]], target_per_cat: int, scenarios: List[str]) -> Dict[str, List[Dict[str, Any]]]:
	out: Dict[str, List[Dict[str, Any]]] = {}
	for cat, arr in inspected.items():
		if len(arr) >= target_per_cat or not arr:
			out[cat] = arr
			continue
		clones: List[Dict[str, Any]] = []
		i = 0
		while len(arr) + len(clones) < target_per_cat:
			src = arr[i % len(arr)]
			scn = scenarios[i % len(scenarios)] if scenarios else f"scenario-{i%3}"
			c = dict(src)
			# tags
			tags = set(c.get("tags") or [])
			tags.add(scn)
			c["tags"] = sorted(tags)
			# ensure unique test id
			base_id = str(c.get("test_id") or f"{cat}-IDX-{i}")
			c["test_id"] = f"{base_id}-V{(i//len(arr))+1}-{scn}"
			clones.append(c)
			i += 1
		out[cat] = arr + clones
	return out


def run_pipeline(xml_dir: str, out_dir: str, policies_path: str, categories: List[str], 
				paraphrase_k: int = 5, min_per_cat: int = 4, max_per_cat: int = None, target_per_cat: int = None,
				use_llm: bool = False, llm_count: int = 5):
	t0=time.time()
	out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

	# 1) Parse
	parser = UniversalParser()
	facts = parser.parse_dir(xml_dir)

	# Load policies bundle once (used by RB/volume boost/LLM)
	try:
		policy_bundle = json.loads(Path(policies_path).read_text(encoding="utf-8"))
	except Exception:
		policy_bundle = {}

	# 2) Generate (Rule-based)
	caps = _capabilities_from_facts(facts)
	gen = RuleBasedGenerator(RuleBasedGeneratorConfig(policies_path=policies_path, min_per_cat=min_per_cat))
	dsl = gen.compile(caps, categories)
	
	# 개수 제한 적용
	if max_per_cat:
		dsl_limited = []
		by_cat = {}
		for item in dsl:
			cat = item["category"]
			by_cat.setdefault(cat, []).append(item)
		
		for cat, items in by_cat.items():
			dsl_limited.extend(items[:max_per_cat])
		dsl = dsl_limited

	# 3) Assemble (Paraphrase → Build → Postprocess → Evidence)
	assembler = TestAssembler(AssembleOptions(base_xml_dir=xml_dir, paraphrase_k=paraphrase_k))
	tests_by_cat = assembler.assemble(facts, dsl)

	# 3.5) LLM Explorer (optional)
	if use_llm:
		try:
			from generators.llm_explorer import LLMExplorer
			print(f"[GIA] LLM Explorer: generating {llm_count} hypotheses...")
			translated = LLMExplorer().from_llm(facts, policy_bundle, n_hypotheses=llm_count)
			print(f"[GIA] LLM Explorer: got {len(translated)} translated items")
			llm_tests = IntentInspector().validate_llm(facts, translated)
			print(f"[GIA] LLM Explorer: validated {len(llm_tests)} tests")
			if llm_tests:
				tests_by_cat.setdefault("LLM_Exploration", []).extend(llm_tests)
				print(f"[GIA] LLM Explorer: added to LLM_Exploration category")
			else:
				print(f"[GIA] LLM Explorer: no tests to add (empty llm_tests)")
		except Exception as e:
			print(f"[GIA] LLM Explorer disabled due to error: {e}")
			import traceback
			traceback.print_exc()

	# 4) (Optional) Inspect — currently passthrough
	inspected = IntentInspector().inspect(tests_by_cat)

	# 4.5) Volume boost (optional)
	if target_per_cat and target_per_cat > 0:
		try:
			scenarios = list(policy_bundle.get("defaults", {}).get("scenarios", []))
		except Exception:
			scenarios = ["운영","감사","야간장애","변경윈도우","점검"]
		inspected = _volume_boost(inspected, target_per_cat, scenarios)

	# Save
	for cat, arr in inspected.items():
		(out / f"tests_{cat}.json").write_text(json.dumps(arr, ensure_ascii=False, indent=2), encoding="utf-8")
	(out / "tests_all.json").write_text(json.dumps(inspected, ensure_ascii=False, indent=2), encoding="utf-8")

	# HTML 요약 리포트 저장
	_write_html_report(out, inspected, "TestSet Report (GIA)")

	# 통계 출력
	total_tests = sum(len(arr) for arr in inspected.values())
	print(f"[GIA] Generated {total_tests} tests:")
	for cat, arr in inspected.items():
		print(f"  - {cat}: {len(arr)} tests")
	
	dt=time.time()-t0
	print(f"[GIA] Done in {dt:.2f}s → out={out}")
	return inspected


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--xml", default="XML_Data", help="XML 디렉토리")
	ap.add_argument("--out", default="out_gia", help="출력 디렉토리")
	ap.add_argument("--cats", nargs="+", required=True, help="카테고리 목록")
	ap.add_argument("--policies", default=str(Path(__file__).parent/"policies"/"policies.json"))
	ap.add_argument("--k", type=int, default=5, help="패턴 변주 수 (현재 2단계에서는 미사용)")
	ap.add_argument("--min-per-cat", type=int, default=4, help="카테고리별 최소 DSL 개수 (현재 DSL 생성 가이드용)")
	ap.add_argument("--max-per-cat", type=int, help="카테고리별 최대 DSL 개수 (제한 없음)")
	ap.add_argument("--target-per-cat", type=int, help="카테고리별 최종 테스트 타겟 개수(부족하면 시나리오 태그로 복제)")
	ap.add_argument("--use-llm", action="store_true", help="LLM 기반 가설/의도 생성 활성화")
	ap.add_argument("--llm-count", type=int, default=5, help="LLM 가설 생성 개수")
	args = ap.parse_args()

	run_pipeline(args.xml, args.out, args.policies, args.cats, 
				paraphrase_k=args.k, min_per_cat=args.min_per_cat, max_per_cat=args.max_per_cat,
				target_per_cat=args.target_per_cat, use_llm=args.use_llm, llm_count=args.llm_count) 