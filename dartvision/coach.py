"""
AI Coach - GenAI feedback grounded in computed statistics.

How it works
1. The numbers (hit rate, bias, spread, trend ...) are computed in Python from the throws.
2. A rule-based draft is always built from those numbers.
3. If a GenAI provider is configured, the LLM rewrites the feedback in natural language
   (chosen tone and language), returning JSON that follows a fixed schema.
4. Every number in the LLM answer is checked against the stats table. If a number cannot be
   verified, the LLM gets one retry; if it still fails, the app shows the rule-based feedback.
5. Every generation is logged to results/coach_log.jsonl so grounding can be evaluated.

Providers (set in the environment or in a .env file in the project folder)
    GEMINI_API_KEY=...            -> Google Gemini (default model gemini-3.8-flash)
    DARTVISION_GEMINI_MODEL=...   -> optional, another Gemini model id
    DARTVISION_LLM=ollama         -> local Ollama instead (DARTVISION_LLM_MODEL, default llama3.2)
    DARTVISION_LLM=off            -> force rule-based feedback
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .board import LABEL_NAMES
from .players import bias_description, player_summary, session_trend

TRIPLE_WIDTH_MM = 8.0  # 99 mm to 107 mm
LOG_PATH = Path(__file__).resolve().parent.parent / "results" / "coach_log.jsonl"

LANGUAGES = {"English": "English", "Hindi": "Hindi (Devanagari script)", "Punjabi": "Punjabi (Gurmukhi script)"}
TONES = {
    "friendly": "a warm, encouraging coach talking to an amateur player",
    "technical": "a precise performance analyst writing for a serious player",
    "short": "a direct coach who uses short, punchy sentences",
}

FEEDBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "description": "2-3 sentences on overall performance and trend"},
        "weakness": {"type": "string", "description": "2-3 sentences on the main weakness, quoting the mm figures"},
        "drill_title": {"type": "string", "description": "Short name of the practice routine"},
        "drill": {"type": "string", "description": "2-3 sentences describing one concrete drill"},
        "target": {"type": "string", "description": "One measurable target for the next session"},
        "motivation": {"type": "string", "description": "One short motivating sentence"},
    },
    "required": ["verdict", "weakness", "drill_title", "drill", "target", "motivation"],
}
ANSWER_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}


# ------------------------------------------------------------------ facts
def player_facts(throws: pd.DataFrame, player: str) -> dict:
    summ = player_summary(throws)
    if player not in set(summ["player"]):
        raise KeyError(player)
    s = summ[summ["player"] == player].iloc[0]
    trend = session_trend(throws)
    t = trend[trend["player"] == player].sort_values("session")
    pt = throws[throws["player"] == player]
    off = pt[pt["label"] != s["aim"]]["label"].value_counts()
    first, last = t.iloc[0], t.iloc[-1]
    best = t.loc[t["avg_score"].idxmax()]
    bias_mm = float(np.hypot(s["bias_x_mm"], s["bias_y_mm"]))
    return {
        "player": player,
        "aim": s["aim"],
        "throws": int(s["throws"]),
        "sessions": int(len(t)),
        "hit_rate_pct": round(float(s["hit_rate_target"]) * 100, 1),
        "hits": int(round(float(s["hit_rate_target"]) * int(s["throws"]))),
        "three_dart_avg": round(float(s["three_dart_avg"]), 1),
        "first_session_3da": round(float(first["avg_score"]) * 3, 1),
        "last_session_3da": round(float(last["avg_score"]) * 3, 1),
        "best_session": int(best["session"]),
        "best_session_3da": round(float(best["avg_score"]) * 3, 1),
        "first_session_error_mm": round(float(first["mean_error_mm"]), 1),
        "last_session_error_mm": round(float(last["mean_error_mm"]), 1),
        "mean_error_mm": round(float(s["mean_error_mm"]), 1),
        "spread_mm": round(float((s["spread_x_mm"] + s["spread_y_mm"]) / 2), 1),
        "bias_x_mm": round(float(s["bias_x_mm"]), 1),
        "bias_y_mm": round(float(s["bias_y_mm"]), 1),
        "bias_mm": round(bias_mm, 1),
        "bias": bias_description(s["bias_x_mm"], s["bias_y_mm"]),
        "miss_rate_pct": round(float(s["miss_rate"]) * 100, 1),
        "top_off_target": off.index[0] if len(off) else None,
        "top_off_target_pct": round(float(off.iloc[0]) / len(pt) * 100, 1) if len(off) else 0.0,
        "triple_width_mm": TRIPLE_WIDTH_MM,
    }


def session_table(throws: pd.DataFrame, player: str) -> list[dict]:
    t = session_trend(throws)
    t = t[t["player"] == player].sort_values("session")
    return [{"session": int(r.session), "three_dart_avg": round(r.avg_score * 3, 1),
             "hit_rate_pct": round(r.hit_rate * 100, 1), "mean_error_mm": round(r.mean_error_mm, 1)}
            for r in t.itertuples()]


def _signed(v: float) -> str:
    return f"+{v:.1f}" if v >= 0 else f"{v:.1f}"


# ------------------------------------------------------------------ rule-based draft
def rule_based_feedback(f: dict) -> dict:
    """Builds feedback from the numbers only. Also adds derived targets to f (used by the LLM)."""
    aim_name = LABEL_NAMES.get(f["aim"], f["aim"])
    delta = round(f["last_session_3da"] - f["first_session_3da"], 1)
    f["delta_3da"] = delta
    trend_word = "improved" if delta > 1 else ("dropped" if delta < -1 else "held steady")
    verdict = (f"Across {f['sessions']} sessions your 3-dart average {trend_word} from "
               f"{f['first_session_3da']} to {f['last_session_3da']} ({_signed(delta)}). "
               f"You hit {aim_name} with {f['hit_rate_pct']}% of {f['throws']} darts; "
               f"your best session was session {f['best_session']} at {f['best_session_3da']}.")

    off = f["top_off_target"]
    off_txt = (f" The most common off-target region is {LABEL_NAMES.get(off, off)} "
               f"({f['top_off_target_pct']}% of all darts).") if off else ""
    if f["bias_mm"] >= 3.0:
        weakness = (f"Your darts land on average {f['bias_mm']} mm {f['bias']} of the {f['aim']} centre "
                    f"({_signed(f['bias_x_mm'])} mm horizontal, {_signed(f['bias_y_mm'])} mm vertical). "
                    f"Grouping spread is {f['spread_mm']} mm per axis.{off_txt}")
        target = round(f["bias_mm"] / 2, 1)
        f["bias_target_mm"] = target
        drill_title = "Aim-correction routine"
        drill = (f"3 sets of 30 darts at {f['aim']}. After each set, note where the group centre sits and "
                 f"adjust your aim point to counter the {f['bias']} drift. "
                 f"Target for next session: reduce the offset from {f['bias_mm']} mm to under {target} mm.")
        target_txt = f"Offset {f['bias_mm']} mm → under {target} mm"
    else:
        weakness = (f"There is no strong directional bias ({f['bias_mm']} mm offset), so the limit is grouping: "
                    f"spread is {f['spread_mm']} mm per axis, while the treble bed is only "
                    f"{f['triple_width_mm']} mm deep.{off_txt}")
        target = round(min(f["hit_rate_pct"] + 5, 100), 1)
        f["hit_target_pct"] = target
        drill_title = "Grouping routine"
        drill = (f"10 rounds of 9 darts at {f['aim']}, counting darts inside the bed each round. "
                 f"Keep the same stance and release point for every dart. "
                 f"Target for next session: raise the hit rate from {f['hit_rate_pct']}% to {target}%.")
        target_txt = f"Hit rate {f['hit_rate_pct']}% → {target}%"
    return {"verdict": verdict, "weakness": weakness, "drill_title": drill_title,
            "drill": drill, "target": target_txt,
            "motivation": "Small, measured changes each session add up — keep logging your throws."}


# ------------------------------------------------------------------ verification
_DIGITS = str.maketrans("०१२३४५६७८९੦੧੨੩੪੫੬੭੮੯", "01234567890123456789")  # Devanagari + Gurmukhi digits
_NUM = re.compile(r"(?<![\w.])([+-]?\d+(?:\.\d+)?)\s*(%|mm)?")


def _allowed_values(facts: dict, extra: list[dict] | None = None) -> list[float]:
    vals = [abs(float(v)) for v in facts.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    for row in extra or []:
        vals += [abs(float(v)) for v in row.values() if isinstance(v, (int, float))]
    return vals


def verify_numbers(text: str, facts: dict, extra: list[dict] | None = None) -> tuple[bool, list[str]]:
    """Every decimal number, percentage or mm value must match a value in the stats (±0.15)."""
    allowed = _allowed_values(facts, extra)
    bad = []
    for num, unit in _NUM.findall(text.translate(_DIGITS)):
        if "." not in num and not unit:
            continue  # plain counts such as "3 sets of 30 darts" or "session 4"
        v = abs(float(num))
        if not any(abs(v - a) <= 0.15 for a in allowed):
            bad.append(num + (unit or ""))
    return (len(bad) == 0, bad)


# ------------------------------------------------------------------ LLM providers
def llm_config() -> dict:
    """Which GenAI provider is active, based on environment variables."""
    choice = os.getenv("DARTVISION_LLM", "").strip().lower()
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if choice in ("off", "none", "rule", "rules"):
        return {"provider": None, "model": None, "reason": "GenAI switched off (DARTVISION_LLM=off)"}
    if choice == "ollama":
        return {"provider": "ollama", "model": os.getenv("DARTVISION_LLM_MODEL", "llama3.2"), "reason": None}
    if choice in ("", "gemini") and key:
        return {"provider": "gemini", "model": os.getenv("DARTVISION_GEMINI_MODEL", "gemini-3.8-flash"), "reason": None}
    return {"provider": None, "model": None,
            "reason": "No GEMINI_API_KEY found - add it to a .env file to enable GenAI feedback"}


def _call_llm(system: str, user: str, schema: dict) -> tuple[str, str, str]:
    cfg = llm_config()
    if cfg["provider"] == "gemini":
        from google import genai  # pip install google-genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        resp = client.models.generate_content(
            model=cfg["model"],
            contents=user,
            config={
                "system_instruction": system,
                "response_mime_type": "application/json",
                "response_json_schema": schema,
                "temperature": 0.4,
            },
        )
        return resp.text, "Gemini", cfg["model"]
    if cfg["provider"] == "ollama":
        import ollama  # pip install ollama
        resp = ollama.chat(model=cfg["model"], format=schema, options={"temperature": 0.4},
                           messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return resp["message"]["content"], "Ollama", cfg["model"]
    raise RuntimeError(cfg["reason"])


def _system_prompt(language: str, tone: str) -> str:
    lang = LANGUAGES.get(language, "English")
    voice = TONES.get(tone, TONES["friendly"])
    return (
        f"You are DartVision Coach, {voice}. Write in {lang}.\n"
        "STRICT RULES:\n"
        "1. Use ONLY the numbers given in the DATA JSON. Never invent, estimate, convert or round numbers.\n"
        "2. Copy every number exactly as written in the data, using Western digits 0-9 (e.g. 8.2 mm, 23.0%).\n"
        "3. Do not claim biomechanical causes (elbow, wrist, release timing) as facts; you may suggest them only as things to try.\n"
        "4. Region codes: T = treble, D = double, S = single, IB = inner bull, OB = outer bull.\n"
        "5. Return JSON that matches the schema. No markdown."
    )


def _log(entry: dict) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry["time"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass  # logging must never break the app


# ------------------------------------------------------------------ public API
def coach_feedback(throws: pd.DataFrame, player: str, language: str = "English", tone: str = "friendly") -> dict:
    facts = player_facts(throws, player)
    rb = rule_based_feedback(facts)          # also adds derived targets to facts
    result = {**rb, "mode": "Rule-based", "provider": None, "model": None, "facts": facts,
              "note": None, "attempts": 0, "latency_ms": None, "language": "English"}
    cfg = llm_config()
    if cfg["provider"]:
        system = _system_prompt(language, tone)
        user = ("DATA (computed from the player's throws):\n" + json.dumps(facts, ensure_ascii=False) +
                "\n\nRule-based draft you may improve on (same numbers):\n" + json.dumps(rb, ensure_ascii=False) +
                "\n\nWrite personalised coaching feedback for this player.")
        t0 = time.perf_counter()
        bad: list[str] = []
        for attempt in (1, 2):
            try:
                prompt = user if attempt == 1 else (
                    user + f"\n\nYour previous answer used numbers that are NOT in the data: {bad}. "
                           "Rewrite it using only numbers copied from the data.")
                text, provider, model = _call_llm(system, prompt, FEEDBACK_SCHEMA)
                out = json.loads(text)
                body = " ".join(str(out.get(k, "")) for k in FEEDBACK_SCHEMA["required"])
                ok, bad = verify_numbers(body, facts)
                result["attempts"] = attempt
                if ok:
                    result.update({k: str(out[k]) for k in FEEDBACK_SCHEMA["required"]})
                    result.update(mode=f"{provider} · {model}", provider=provider, model=model, language=language)
                    break
            except Exception as e:  # noqa: BLE001 - any provider/network/JSON error falls back safely
                result["note"] = f"GenAI unavailable ({type(e).__name__}: {str(e)[:120]}). Showing rule-based feedback."
                bad = []
                break
        else:
            result["note"] = f"The AI answer contained unverified numbers {bad} twice, so the rule-based feedback is shown."
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000)
        _log({"kind": "feedback", "player": player, "provider": cfg["provider"], "model": cfg["model"],
              "language": language, "tone": tone, "attempts": result["attempts"],
              "used_llm": result["provider"] is not None, "unverified": bad,
              "error": result["note"] if result["note"] and "unavailable" in result["note"] else None,
              "latency_ms": result["latency_ms"]})
    else:
        result["note"] = cfg["reason"]
    ok, bad = verify_numbers(" ".join(str(result[k]) for k in FEEDBACK_SCHEMA["required"]), facts)
    result["verified"], result["unverified"] = ok, bad
    return result


def answer_question(throws: pd.DataFrame, player: str, question: str, language: str = "English") -> dict:
    """Answers from the data. Common questions are computed directly; anything else goes to the LLM
    (grounded in the stats and verified). Returns {"answer", "source"}."""
    q = question.lower()
    f = player_facts(throws, player)
    trend = session_trend(throws)
    t = trend[trend["player"] == player]
    summ = player_summary(throws)

    m = re.search(r"player\s+([a-e])\b", q)
    other = None
    if m:
        cand = [p for p in summ["player"] if p.lower().startswith(f"player {m.group(1)}")]
        other = cand[0] if cand and cand[0] != player else None

    def data(text):
        return {"answer": text, "source": "data"}

    if other and any(w in q for w in ("compare", "vs", "than", "versus", "spread", "better")):
        o = player_facts(throws, other)
        return data(f"{player}: spread {f['spread_mm']} mm, mean error {f['mean_error_mm']} mm, "
                    f"3-dart average {f['three_dart_avg']}. {other}: spread {o['spread_mm']} mm, "
                    f"mean error {o['mean_error_mm']} mm, 3-dart average {o['three_dart_avg']}.")
    if "best" in q and "session" in q:
        if any(w in q for w in ("hit", "bull", "target", "accura")):
            b = t.loc[t["hit_rate"].idxmax()]
            return data(f"Session {int(b['session'])} had your best hit rate on {f['aim']}: "
                        f"{b['hit_rate'] * 100:.1f}% (3-dart average {b['avg_score'] * 3:.1f}).")
        b = t.loc[t["avg_score"].idxmax()]
        return data(f"Session {int(b['session'])} was your best by score: 3-dart average {b['avg_score'] * 3:.1f}.")
    if "worst" in q and "session" in q:
        b = t.loc[t["avg_score"].idxmin()]
        return data(f"Session {int(b['session'])} was your lowest: 3-dart average {b['avg_score'] * 3:.1f}.")
    if "miss" in q:
        off = f["top_off_target"]
        return data(f"Miss rate (off the board): {f['miss_rate_pct']}%. Darts that missed {f['aim']} most often "
                    f"landed in {LABEL_NAMES.get(off, off)} ({f['top_off_target_pct']}% of all darts).")
    if any(w in q for w in ("bias", "tendency", "drift")):
        return data(f"Average offset from {f['aim']}: {f['bias_mm']} mm ({f['bias']}); "
                    f"{f['bias_x_mm']} mm horizontal, {f['bias_y_mm']} mm vertical.")

    # Anything else: ask the LLM, grounded in the stats + per-session table, and verify it.
    cfg = llm_config()
    sessions = session_table(throws, player)
    if cfg["provider"]:
        system = _system_prompt(language, "friendly") + (
            "\n6. If the DATA cannot answer the question, say so and suggest what data would be needed.")
        user = ("DATA:\n" + json.dumps({"player_stats": f, "sessions": sessions}, ensure_ascii=False) +
                f"\n\nQUESTION: {question}\nAnswer in at most 3 sentences.")
        try:
            text, provider, model = _call_llm(system, user, ANSWER_SCHEMA)
            ans = str(json.loads(text)["answer"])
            ok, bad = verify_numbers(ans, f, sessions)
            _log({"kind": "question", "player": player, "provider": cfg["provider"], "model": cfg["model"],
                  "attempts": 1, "used_llm": ok, "unverified": bad, "error": None})
            if ok:
                return {"answer": ans, "source": f"{provider} · {model}"}
        except Exception:  # noqa: BLE001
            pass
    if any(w in q for w in ("average", "score", "improve", "progress", "trend")):
        return data(f"3-dart average {f['three_dart_avg']} overall; session 1: {f['first_session_3da']}, "
                    f"session {f['sessions']}: {f['last_session_3da']}.")
    if any(w in q for w in ("hit", "accura", "rate")):
        return data(f"You hit {f['aim']} with {f['hit_rate_pct']}% of darts ({f['hits']} of {f['throws']}).")
    return data("I can answer from your data about: best or worst session, hit rate, misses, bias/tendency, "
                "score trend, or a comparison with another player (e.g. 'compare my spread to Player A'). "
                "Add a GEMINI_API_KEY to ask free-form questions.")


def coach_stats() -> dict:
    """Grounding evaluation of all logged GenAI generations."""
    rows = []
    if LOG_PATH.exists():
        for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    llm_rows = [r for r in rows if not r.get("error")]
    fb = [r for r in llm_rows if r.get("kind") == "feedback"]
    lat = [r["latency_ms"] for r in fb if r.get("latency_ms") is not None]
    return {
        "generations": len(llm_rows),
        "verified": sum(1 for r in llm_rows if r.get("used_llm")),
        "first_try": sum(1 for r in fb if r.get("used_llm") and r.get("attempts") == 1),
        "after_retry": sum(1 for r in fb if r.get("used_llm") and r.get("attempts") == 2),
        "rejected": sum(1 for r in llm_rows if not r.get("used_llm")),
        "errors": sum(1 for r in rows if r.get("error")),
        "mean_latency_ms": round(float(np.mean(lat))) if lat else 0,
    }