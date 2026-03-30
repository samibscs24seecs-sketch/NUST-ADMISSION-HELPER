"""
NUST Chatbot Chain

Architecture:
  - The LLM is the primary responder for all factual questions.
  - Retrieved facts are injected DIRECTLY into the user message (not the system
    prompt) so phi3:mini reads them instead of relying on training-data hallucination.
  - Short-circuits exist ONLY for operations the LLM cannot do reliably:
      * Merit calculation  (arithmetic — always correct)
      * Eligibility check  (arithmetic — always correct)
      * Off-topic guard    (security)
      * Casual replies     (no factual content needed)
      * Offline fallback   (Ollama not running)
"""

import json
import logging
import re
from pathlib import Path
import sys
from typing import AsyncGenerator

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from chatbot.llm import OllamaLLM, llm as default_llm
from rag.retriever import NUSTRetriever, retriever as default_retriever

logger = logging.getLogger("NUSTChain")


# ─────────────────────────────────────────────────────────────
#  STATIC DATA
# ─────────────────────────────────────────────────────────────

def _load_cutoffs_data() -> list[dict]:
    with open(ROOT / "data" / "cutoffs.json", encoding="utf-8") as f:
        return json.load(f)["cutoffs"]

def _load_fees_data() -> dict:
    with open(ROOT / "data" / "fees.json", encoding="utf-8") as f:
        return json.load(f)

_CUTOFFS_DATA   = _load_cutoffs_data()
_CUTOFFS        = {f"{i['name']} ({i['school']})": i["cutoff"] for i in _CUTOFFS_DATA}
_CUTOFFS_SIMPLE = {i["name"]: i["cutoff"] for i in _CUTOFFS_DATA}
_FEES_DATA      = _load_fees_data()

# Inline cutoff table injected into every LLM prompt so the model never guesses
_CUTOFF_TABLE = "\n".join(
    f"  {i['name']} ({i['school']}): {i['cutoff']}%"
    for i in sorted(_CUTOFFS_DATA, key=lambda x: x["cutoff"], reverse=True)
)


# ─────────────────────────────────────────────────────────────
#  SYSTEM PROMPT  — kept short so phi3:mini has room to answer
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are NUSTBot, a data-driven NUST admissions assistant.

STRICT RULES — follow every rule on every response:
1. Use ONLY the cutoffs and information given to you in each message.
   Do NOT use outside training data for any NUST-specific fact (fees, cutoffs, dates, programmes, processes).
2. If a fee or merit number is NOT explicitly in the provided data, do NOT estimate it.
   Say: "I don't have that exact figure — please check ugadmissions.nust.edu.pk for current fees."
3. Never invent programme names, URLs, statistics, rankings, or partnerships.
4. Answer directly — give the fact first, then any explanation.
5. Do not say "based on the provided information" or "according to the context" — just answer.
6. NUST entry test = NET. Never say NEET (Indian exam, unrelated to NUST Pakistan).
7. Write ONE single cohesive response. Do NOT repeat any section heading or bullet header more than once. Use a clean Markdown list — not repeated headers for each point."""


# ─────────────────────────────────────────────────────────────
#  SHORT-CIRCUIT: MERIT CALCULATION  (arithmetic, never LLM)
# ─────────────────────────────────────────────────────────────

_MERIT_CALC_TRIGGER = re.compile(
    r'\b(calculat|calc|my merit|my aggregate|what.*aggregate|aggregate.*calculat|'
    r'compute|work.*out|figure.*out)\b',
    re.I,
)
_NET_VAL    = re.compile(r'\bnet\b[^/\d]*?(\d+(?:\.\d+)?)\b', re.I)
_FSC_VAL    = re.compile(r'\bfsc\b[^/\d]*?(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', re.I)
_MATRIC_VAL = re.compile(r'\bmatric\b[^/\d]*?(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', re.I)


def _build_merit_reply(net: float, fsc: float, fsc_t: float,
                       mat: float, mat_t: float) -> str:
    net_c = (net / 200) * 75
    fsc_c = (fsc / fsc_t) * 15
    mat_c = (mat / mat_t) * 10
    agg   = net_c + fsc_c + mat_c
    eligible   = [(p, c) for p, c in _CUTOFFS.items() if agg >= c]
    ineligible = [(p, c) for p, c in _CUTOFFS.items() if agg < c]
    elig_lines = "\n".join(
        f"  ✅ {p} (cutoff {c}%)" for p, c in
        sorted(eligible, key=lambda x: x[1], reverse=True)
    ) or "  ❌ None with current scores"
    close_str = ", ".join(
        f"{p} ({c}%)" for p, c in sorted(ineligible, key=lambda x: x[1])[:4]
    )
    return (
        f"**Your NUST Merit Calculation**\n\n"
        f"| Component | Marks | Contribution |\n"
        f"|-----------|-------|--------------|\n"
        f"| NET       | {net:.0f}/200     | × 75 = **{net_c:.2f}** |\n"
        f"| FSc       | {fsc:.0f}/{fsc_t:.0f} | × 15 = **{fsc_c:.2f}** |\n"
        f"| Matric    | {mat:.0f}/{mat_t:.0f} | × 10 = **{mat_c:.2f}** |\n\n"
        f"**Aggregate = {net_c:.2f} + {fsc_c:.2f} + {mat_c:.2f} = {agg:.2f}%**\n\n"
        f"**Programmes you qualify for ({len(eligible)}):**\n{elig_lines}\n"
        + (f"\n*Just out of reach: {close_str}*" if close_str else "")
    )


def _try_merit_calc(message: str) -> str | None:
    if not _MERIT_CALC_TRIGGER.search(message):
        return None
    net_m = _NET_VAL.search(message)
    fsc_m = _FSC_VAL.search(message)
    mat_m = _MATRIC_VAL.search(message)
    if not (net_m and fsc_m and mat_m):
        return None
    try:
        net   = float(net_m.group(1))
        fsc   = float(fsc_m.group(1));  fsc_t = float(fsc_m.group(2))
        mat   = float(mat_m.group(1));  mat_t = float(mat_m.group(2))
        if not (0 <= net <= 200 and 0 < fsc <= fsc_t and 0 < mat <= mat_t):
            return None
        return _build_merit_reply(net, fsc, fsc_t, mat, mat_t)
    except (ValueError, ZeroDivisionError):
        return None


# ─────────────────────────────────────────────────────────────
#  SHORT-CIRCUIT: ELIGIBILITY CHECK  (arithmetic, never LLM)
# ─────────────────────────────────────────────────────────────

# Matches first-person eligibility questions AND questions where the user mentions
# their own aggregate/score — these must ALWAYS be answered by Python arithmetic,
# never by the LLM (which makes comparison errors on small models).
# General FAQs ("Can Pre-Medical students apply?") still go to RAG/LLM.
_ELIGIBILITY_PATTERN = re.compile(
    r'\b('
    r'am\s+i\s+(eligible|qualified)|'
    r'will\s+i\s+(get|qualify|be\s+selected)|'
    r'do\s+i\s+qualify|'
    r'can\s+i\s+(get\s+into|get\s+admission|secure\s+admission)|'
    r'what\s+are\s+my\s+(chances|odds)|'
    r'programs?\s+for\s+me|'
    r'my\s+options?|'
    r'which\s+programs?\s+(can\s+i|should\s+i|am\s+i)|'
    r'with\s+my\s+(aggregate|marks|score|percent)'
    r')\b',
    re.I,
)

# Extract a percentage aggregate directly from the user message
# e.g. "I have 79%" / "my aggregate is 79.5%" / "aggregate: 79"
_AGG_IN_MESSAGE = re.compile(
    r'(?:aggregate|agg|score|percent|%|marks)[^\d]*(\d{2,3}(?:\.\d+)?)\s*%?'
    r'|(\d{2,3}(?:\.\d+)?)\s*%\s*(?:aggregate|agg|score)?',
    re.I,
)

_NO_SCORES_REPLY = (
    "I need your scores to check eligibility. Please share:\n\n"
    "- **NET score** (out of 200)\n"
    "- **FSc marks / total** (e.g. 1000/1100)\n"
    "- **Matric marks / total** (e.g. 950/1100)\n\n"
    "I will calculate your aggregate and list every programme you qualify for."
)


def _extract_aggregate_from_message(message: str) -> float | None:
    """Try to read a percentage aggregate stated directly in the user message."""
    m = _AGG_IN_MESSAGE.search(message)
    if m:
        raw = m.group(1) or m.group(2)
        try:
            val = float(raw)
            if 40.0 <= val <= 100.0:   # sanity check: valid aggregate range
                return val
        except (TypeError, ValueError):
            pass
    return None


def _build_eligibility_reply(aggregate: float) -> str:
    # Simple Python comparison — aggregate >= cutoff means eligible. Always correct.
    eligible = sorted(
        [(n, c) for n, c in _CUTOFFS_SIMPLE.items() if aggregate >= c],
        key=lambda x: x[1], reverse=True,
    )
    close = sorted(
        [(n, c) for n, c in _CUTOFFS_SIMPLE.items() if 0 < c - aggregate <= 5],
        key=lambda x: x[1],
    )
    if eligible:
        lines = [f"With aggregate **{aggregate:.2f}%** you qualify for **{len(eligible)} programme(s)**:\n"]
        for name, co in eligible:
            margin = aggregate - co
            lines.append(f"- **{name}** — cutoff {co}% | +{margin:.1f}% above cutoff | "
                         + ("Likely" if margin >= 2 else "Borderline"))
    else:
        lines = [f"Aggregate **{aggregate:.2f}%** does not meet any cutoff in our list."]
    if close:
        lines.append("\n**Just out of reach (within 5%):**")
        for name, co in close:
            lines.append(f"- {name} — need {co - aggregate:.1f}% more (cutoff {co}%)")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  SHORT-CIRCUIT: FORMULA  (LLM cannot reproduce math notation reliably)
# ─────────────────────────────────────────────────────────────

_FORMULA_PATTERN = re.compile(
    r'\b(formula|how.*calculat|aggregate.*how|merit.*formula|explain.*aggregate|'
    r'aggregate.*explain|worked\s+example|how.*aggregate.*work)\b',
    re.I,
)

FORMULA_REPLY = """\
**NUST Aggregate Formula**

```
Aggregate = (NET / 200 × 75) + (FSc / FSc_Total × 15) + (Matric / Matric_Total × 10)
```

| Component | Weight | Max Points |
|-----------|--------|------------|
| NET (out of 200) | 75% | 75 |
| FSc / Intermediate | 15% | 15 |
| Matric / SSC | 10% | 10 |

**Worked Example:**
- NET = 160/200 → 160/200 × 75 = **60.00**
- FSc = 1000/1100 → 1000/1100 × 15 = **13.64**
- Matric = 950/1100 → 950/1100 × 10 = **8.64**
- **Aggregate = 60.00 + 13.64 + 8.64 = 82.27%**

No Hafiz bonus. No extra-curricular bonus. SAT ≥ 1100 accepted instead of NET. NUST uses your best NET score."""

_FEE_PATTERN = re.compile(
    r'\b(fee|fees|tuition|cost|how\s+much|price|expense|charges?|'
    r'semester\s+fee|annual\s+fee|total\s+cost|lakh|lac|rupee|pkr)\b',
    re.I,
)

_SCHOOL_FEE_KEYWORDS = {
    "seecs": "SEECS", "electrical": "SEECS", "computer": "SEECS",
    "cs": "SEECS", "ai": "SEECS", "software": "SEECS",
    "smme": "SMME", "mechanical": "SMME", "mechatronics": "SMME",
    "scee": "SCEE", "civil": "SCEE", "environmental": "SCEE",
    "scme": "SCME", "chemical": "SCME", "materials": "SCME",
    "nbs": "NBS", "business": "NBS", "bba": "NBS", "accounting": "NBS",
    "sada": "SADA", "architecture": "SADA",
    "asab": "ASAB", "bio": "ASAB", "biotechnology": "ASAB",
    "sns": "SNS", "mathematics": "SNS", "physics": "SNS", "chemistry": "SNS",
    "s3h": "S3H", "economics": "S3H", "psychology": "S3H",
    "igis": "IGIS", "gis": "IGIS",
}


def _build_fee_reply(message: str) -> str:
    """Return fee info from fees.json. Never estimates — always redirects for verification."""
    msg_lower = message.lower()
    school = next((code for kw, code in _SCHOOL_FEE_KEYWORDS.items() if kw in msg_lower), None)
    per_sem = _FEES_DATA.get("per_semester_pkr", {})
    additional = _FEES_DATA.get("additional_per_semester_pkr", {})
    one_time = _FEES_DATA.get("one_time_pkr", {})
    hostel = _FEES_DATA.get("hostel_per_month_pkr", {})
    verify_url = _FEES_DATA.get("_verify_at", "ugadmissions.nust.edu.pk")
    note = _FEES_DATA.get("_note", "")

    if school and school in per_sem:
        entry = per_sem[school]
        lines = [
            f"**{school} Approximate Semester Fee:** {entry['display']} ({entry['note']})\n",
            f"Additional per semester:",
            f"- Student Activity Fund: PKR {additional.get('student_activity_fund', 'N/A')}",
            f"- Library fee: PKR {additional.get('library_fee', 'N/A')}",
            f"- Medical fund: PKR {additional.get('medical_fund', 'N/A')}",
            f"\nOne-time at admission:",
            f"- Security deposit (refundable): PKR {one_time.get('security_deposit_refundable', 'N/A')}",
            f"\nHostel (if applicable): PKR {hostel.get('range', 'N/A')}/month — {hostel.get('note', '')}",
            f"\n> **These are approximate 2025-26 figures.** Fees are revised annually.",
            f"> Always verify the current amount at: **{verify_url}**",
        ]
    else:
        rows = "\n".join(
            f"  {code}: {v['display']} — {v['note']}"
            for code, v in per_sem.items()
        )
        total = _FEES_DATA.get("total_degree_estimate_pkr", {})
        lines = [
            "**NUST Approximate Semester Fees (2025-26)**\n",
            rows,
            f"\nEngineering degree (8 semesters): {total.get('engineering_8sem', 'N/A')}",
            f"\n> **These are approximate figures.** Fees change each year.",
            f"> Verify exact current fees at: **{verify_url}**",
        ]

    return "\n".join(lines)




# ─────────────────────────────────────────────────────────────
#  SHORT-CIRCUIT: OFF-TOPIC / INJECTION GUARD
# ─────────────────────────────────────────────────────────────

_OFF_TOPIC_PATTERN = re.compile(
    r'\b(write\s+(a|an|me|for|the|it)|create\s+(a|an|me)|generate|compose|draft|'
    r'essay|narrative|plan\s+of\s+action|new\s+york\s+times|instructional|'
    r'constraints:|document\s+type|dissect|verbatim|'
    r'without\s+(directly\s+)?quot|your\s+task\s*[:is]|'
    r'from\s+the\s+perspective\s+of|fbi.style)\b',
    re.I,
)

_OFF_TOPIC_REPLY = (
    "I can only answer questions about NUST admissions — programmes, cutoffs, "
    "merit calculation, NET exam, and eligibility. What would you like to know?"
)


# ─────────────────────────────────────────────────────────────
#  SHORT-CIRCUIT: CASUAL / SMALL-TALK
# ─────────────────────────────────────────────────────────────

_CASUAL_PATTERNS = [
    (re.compile(r'^\s*(hi|hello|hey|salam|assalam|aoa|howdy)\b', re.I),
     "Hello! I'm NUSTBot. Ask me anything about NUST admissions, programmes, NET, or merit calculation."),
    (re.compile(r'\b(how are you|how r u|you okay|you good)\b', re.I),
     "Doing great! What can I help you with regarding NUST admissions?"),
    (re.compile(r'^\s*(ok|okay|got it|understood|noted|sure|alright|yes|yep|yeah|no|nope)\s*[.!]*\s*$', re.I),
     "Got it! What else would you like to know about NUST?"),
    (re.compile(r'\b(thanks|thank you|thx|ty|great|perfect|awesome|appreciate)\b', re.I),
     "Glad I could help! Feel free to ask anything else about NUST admissions."),
]


def _get_casual_reply(message: str) -> str | None:
    if len(message.strip()) < 60:
        for pattern, reply in _CASUAL_PATTERNS:
            if pattern.search(message):
                return reply
    return None


# ─────────────────────────────────────────────────────────────
#  FALLBACK
# ─────────────────────────────────────────────────────────────

FALLBACK_MESSAGE = (
    "I can't reach the AI model right now. Key facts:\n\n"
    "- **Formula:** Aggregate = (NET/200×75) + (FSc/Total×15) + (Matric/Total×10)\n"
    "- No Hafiz bonus, no reserved seats for general applicants\n"
    "- SAT ≥ 1100 accepted instead of NET\n"
    "- Apply at: ugadmissions.nust.edu.pk\n\n"
    "Start Ollama (`ollama serve`) for full responses."
)


# ─────────────────────────────────────────────────────────────
#  STOP SENTINELS  (catch system prompt echo from phi3:mini)
# ─────────────────────────────────────────────────────────────

_STOP_SENTINELS = [
    "YOU ARE NUSTBOT", "SYSTEM PROMPT", "ANSWER THE STUDENT",
    "NEVER INVENT", "NUST ENTRY TEST =",
    # Catch the model referencing the context block instead of just answering
    "PROVIDED FACTS", "THE FACTS", "AS PER FACTS", "FACTS MESSAGE",
    "KNOWN CUTOFFS:", "RELEVANT INFORMATION:", "QUESTION:",
    # Catch meta-commentary the model sometimes emits
    "AS PER YOUR REQUEST", "BASED ON THE INFORMATION PROVIDED",
    "ACCORDING TO THE CONTEXT", "THE CONTEXT BLOCK",
]

# Fake URL pattern — phi3:mini sometimes invents long garbled URLs
_FAKE_URL_PATTERN = re.compile(
    r'https?://(?!(?:ugadmissions\.nust\.edu\.pk|nust\.edu\.pk|nustify\.com|nustrive1\.vercel\.app)'
    r'(?:[/?#]|$))'
    r'\S{30,}',   # any URL over 30 chars that isn't one of our known good domains
    re.I,
)


# ─────────────────────────────────────────────────────────────
#  CHAIN
# ─────────────────────────────────────────────────────────────

def _format_history(history: list[dict]) -> list[dict]:
    messages = []
    for turn in history:
        if u := turn.get("user", "").strip():
            messages.append({"role": "user", "content": u})
        if a := turn.get("assistant", "").strip():
            messages.append({"role": "assistant", "content": a})
    return messages


class NUSTChain:
    """RAG + LLM chain. LLM answers from retrieved facts placed in the user message."""

    def __init__(self, llm: OllamaLLM = None, retriever: NUSTRetriever = None):
        self.llm       = llm or default_llm
        self.retriever = retriever or default_retriever

    def _build_messages(self, message: str, history: list[dict],
                        session: dict, rag_context: str) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add recent history (last 3 turns max to save context)
        for turn in _format_history(history)[-6:]:
            messages.append(turn)

        # Build the user message: facts first, then the question.
        # Placing facts directly before the question is the key change —
        # phi3:mini reads the immediate context reliably, unlike distant system prompt text.
        facts_block = f"KNOWN CUTOFFS:\n{_CUTOFF_TABLE}\n"

        if rag_context:
            facts_block += f"\nRELEVANT INFORMATION:\n{rag_context}\n"

        session_block = ""
        parts = []
        if agg := session.get("aggregate"):
            try: parts.append(f"Student aggregate: {float(agg):.2f}%")
            except (TypeError, ValueError): pass
        if net := session.get("net_score"):
            try: parts.append(f"Student NET score: {float(net)}/200")
            except (TypeError, ValueError): pass
        if parts:
            session_block = "\nSTUDENT SESSION:\n" + "\n".join(parts) + "\n"

        user_content = (
            f"{facts_block}"
            f"{session_block}"
            f"\nQUESTION: {message}"
        )
        messages.append({"role": "user", "content": user_content})
        return messages

    async def stream(
        self,
        message: str,
        history: list[dict] = None,
        session: dict = None,
    ) -> AsyncGenerator[str, None]:
        history = history or []
        session = session or {}

        # ── 1. Injection guard ───────────────────────────────
        if _OFF_TOPIC_PATTERN.search(message):
            yield _OFF_TOPIC_REPLY
            return

        # ── 2. Casual ────────────────────────────────────────
        casual = _get_casual_reply(message)
        if casual:
            yield casual
            return

        # ── 3. Formula (LLM cannot reproduce math notation) ──
        if _FORMULA_PATTERN.search(message):
            yield FORMULA_REPLY
            return

        # ── 3b. Fees — always from fees.json, never LLM-estimated ──
        if _FEE_PATTERN.search(message):
            yield _build_fee_reply(message)
            return

        # ── 4. Merit calculation (arithmetic) ────────────────
        merit_reply = _try_merit_calc(message)
        if merit_reply:
            yield merit_reply
            return

        # ── 4. Eligibility check (arithmetic — never LLM) ────
        if _ELIGIBILITY_PATTERN.search(message):
            # Priority: aggregate stated in message > session aggregate
            agg = _extract_aggregate_from_message(message) or session.get("aggregate")
            if agg:
                try:
                    yield _build_eligibility_reply(float(agg))
                except (TypeError, ValueError):
                    yield _NO_SCORES_REPLY
            else:
                yield _NO_SCORES_REPLY
            return

        # ── 5. Check Ollama ───────────────────────────────────
        health = await self.llm.check_ollama_health()
        if health["status"] != "running":
            yield FALLBACK_MESSAGE
            return

        # ── 6. Retrieve facts from knowledge base ─────────────
        rag_result  = self.retriever.retrieve(message)
        rag_context = rag_result.get("context", "")
        confidence  = rag_result.get("confidence", 0.0)
        logger.debug("RAG confidence: %.1f%% for: %s", confidence, message[:60])

        # ── 7. Build messages and stream ──────────────────────
        messages = self._build_messages(message, history, session, rag_context)

        buf = ""
        full_output = ""
        sentinel_len = max(len(s) for s in _STOP_SENTINELS)

        async for token in self.llm.stream_response(messages):
            buf += token
            buf_upper = buf.upper()

            # 1. Sentinel check — stop if system prompt echo detected
            hit = next((s for s in _STOP_SENTINELS if s.upper() in buf_upper), None)
            if hit:
                cut  = buf_upper.find(hit.upper())
                safe = buf[:cut].rstrip()
                if safe:
                    yield safe
                return

            if len(buf) > sentinel_len:
                to_yield = buf[:-sentinel_len]

                # 2. Fake URL scrub — replace invented URLs with a safe placeholder
                to_yield = _FAKE_URL_PATTERN.sub(
                    "[URL removed — please check ugadmissions.nust.edu.pk]", to_yield
                )

                full_output += to_yield

                # 3. Repetition guard — stop if model starts looping
                # Check multiple window sizes: short (catches header loops),
                # medium, and long (catches paragraph-level repetition).
                if len(full_output) > 200:
                    for window in (50, 80, 120):
                        if len(full_output) > window * 3:
                            tail = full_output[-window:]
                            if full_output[:-window].count(tail) >= 1:
                                yield to_yield
                                return

                yield to_yield
                buf = buf[-sentinel_len:]

        if buf:
            # Final fake URL scrub on remaining buffer
            buf = _FAKE_URL_PATTERN.sub(
                "[URL removed — please check ugadmissions.nust.edu.pk]", buf
            )
            yield buf

    async def get_full_response(
        self,
        message: str,
        history: list[dict] = None,
        session: dict = None,
    ) -> dict:
        history = history or []
        session = session or {}

        health = await self.llm.check_ollama_health()
        if health["status"] != "running":
            return {"response": FALLBACK_MESSAGE, "confidence": 0.0,
                    "sources": [], "ollama_status": "offline"}

        rag_result  = self.retriever.retrieve(message)
        rag_context = rag_result.get("context", "")
        confidence  = rag_result.get("confidence", 0.0)
        sources     = rag_result.get("sources", [])

        messages = self._build_messages(message, history, session, rag_context)
        response = await self.llm.get_full_response(messages)

        return {"response": response, "confidence": confidence,
                "sources": sources, "ollama_status": "running"}


# Singleton
chain = NUSTChain()
