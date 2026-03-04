import os
import json
import re
import pandas as pd
import ollama
from flask import Flask, request, jsonify, render_template, Response, stream_with_context

# ── Config ──────────────────────────────────────────────────────────────────
CSV_PATH  = os.path.join(os.path.dirname(__file__), "Medical-Equipment-Suppliers.csv")
MODEL     = "gemma3:4b"
MAX_ROWS  = 30          # max rows we feed into the answer prompt
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ── Load CSV once at startup ─────────────────────────────────────────────────
print("Loading CSV …")
df = pd.read_csv(CSV_PATH, dtype=str, low_memory=False)
df.columns = [c.strip().lower() for c in df.columns]

# Pre-build small lookup strings to inject into prompts as hints
_states      = sorted(df["practicestate"].dropna().unique().tolist())
_top_supplies = (
    df["supplieslist"].dropna()
      .str.split("|").explode().str.strip()
      .value_counts().head(40).index.tolist()
)
_top_specialities = (
    df["specialitieslist"].dropna()
      .str.split("|").explode().str.strip()
      .value_counts().head(30).index.tolist()
)

STATES_HINT      = ", ".join(_states)
SUPPLIES_HINT    = " | ".join(_top_supplies)
SPECIALTIES_HINT = " | ".join(_top_specialities)

print(f"CSV loaded – {len(df):,} rows, {len(df.columns)} columns.")

# ── State name → 2-letter code map ──────────────────────────────────────────
STATE_NAME_TO_CODE = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}
# ─────────────────────────────────────────────────────────────────────────────


# ── Prompt templates ─────────────────────────────────────────────────────────
FILTER_SYSTEM = f"""You are a data filter assistant for a medical equipment suppliers database.

Columns available:
  provider_id, acceptsassignement (True/False), participationbegindate,
  businessname, practicename, practiceaddress1, practicecity, practicestate (2-letter),
  practicezip9code, telephonenumber, specialitieslist (pipe-separated),
  providertypelist (pipe-separated), supplieslist (pipe-separated),
  latitude, longitude, is_contracted_for_cba (True/False)

Known 2-letter state codes in the data: {STATES_HINT}

Common supply values (pipe-separated, partial list): {SUPPLIES_HINT}

Common speciality values (partial list): {SPECIALTIES_HINT}

CRITICAL RULES — READ CAREFULLY:
1. "state" MUST be a 2-letter uppercase code ("FL" not "Florida", "TX" not "Texas", "NY" not "New York").
2. "city" MUST be UPPERCASE. ONLY set city if the user EXPLICITLY names a specific city. Do NOT guess or infer a city.
3. "zip" ONLY set if the user explicitly mentions a ZIP code. Do NOT guess a ZIP. Set to null otherwise.
4. "supply_keywords" must be SHORT (1-3 words). Use the core word only, NOT full phrases from the data.
   GOOD: "Oxygen", "Wheelchair", "CPAP", "Prosthetic", "Nebulizer", "Glucose", "Ostomy", "Ventilator"
   BAD: "Oxygen Equipment and/or Supplies", "Blood Glucose Monitors/Supplies (Non-Mail Order)"
5. "specialty_keywords" should also be short: "Pharmacy", "Orthotic", "Prosthetic", "Respiratory"
6. Only populate fields that are CLEARLY AND EXPLICITLY stated by the user. Use null/[] for everything not mentioned.
7. NEVER invent or guess location details (city, zip) that are not in the question.
8. If the user mentions a ZIP code, set "zip" to that value and leave "city" as null.

Return ONLY a JSON object with these exact keys:
{{
  "state":              "2-letter state code or null",
  "city":               "CITY NAME IN UPPERCASE — only if explicitly in the question, else null",
  "supply_keywords":    ["short_keyword"] or [],
  "specialty_keywords": ["short_keyword"] or [],
  "name_contains":      "partial business name or null",
  "accepts_assignment": true/false/null,
  "is_contracted_cba":  true/false/null,
  "zip":                "zip prefix string — only if explicitly in the question, else null",
  "limit":              {MAX_ROWS}
}}

Return ONLY the JSON object. No explanation, no markdown fences, no extra text."""


ANSWER_SYSTEM = """You are a helpful assistant that answers questions about medical equipment suppliers.
You will be given a user question and a table of matching supplier records.
Answer clearly and concisely in plain English. 
If the table has useful details, summarise or list them.
If asked for a count, give the count from the table.
Keep your answer focused and avoid repeating every field of every row unless specifically asked."""
# ─────────────────────────────────────────────────────────────────────────────


# ── Filter helpers ───────────────────────────────────────────────────────────
def _col_contains(series: pd.Series, keyword: str) -> pd.Series:
    """Case-insensitive substring match on a Series, handles NaN."""
    return series.fillna("").str.contains(keyword, case=False, regex=False)


def _normalize_state(raw: str) -> str:
    """Convert full state name OR 2-letter code to uppercase 2-letter code."""
    s = raw.strip()
    if len(s) == 2:
        return s.upper()
    return STATE_NAME_TO_CODE.get(s.lower(), s.upper())


def _keyword_mask(series: pd.Series, keyword: str) -> pd.Series:
    """
    Match a keyword against a pipe-delimited series.
    Splits multi-word keywords and requires ALL words to appear (AND logic
    within a keyword, OR logic across keywords is handled by the caller).
    Short words <= 3 chars are matched as whole words to avoid noise.
    """
    words = [w for w in keyword.split() if len(w) > 2]  # skip tiny stop words
    if not words:
        words = keyword.split()
    mask = pd.Series([True] * len(series), index=series.index)
    for word in words:
        mask &= series.fillna("").str.contains(word, case=False, regex=False)
    return mask


def _apply_keyword_filters(result: pd.DataFrame, col: str, keywords: list) -> pd.DataFrame:
    """
    For each keyword apply _keyword_mask and keep rows that match ANY keyword (OR).
    If a keyword produces 0 extra rows on its own, skip it (don't over-constrain).
    """
    if not keywords:
        return result
    combined_mask = pd.Series([False] * len(result), index=result.index)
    matched_any = False
    for kw in keywords:
        m = _keyword_mask(result[col], kw)
        if m.sum() > 0:
            combined_mask |= m
            matched_any = True
    if not matched_any:
        # No keyword matched at all — return unfiltered so other filters still help
        return result
    return result[combined_mask]


def filter_df(filters: dict) -> pd.DataFrame:
    """Apply filters progressively; relax gracefully when results would be 0."""
    print(f"[filter_df] raw filters: {filters}")
    result = df.copy()

    # ── State ──
    if filters.get("state"):
        code = _normalize_state(filters["state"])
        narrowed = result[result["practicestate"].fillna("").str.upper() == code]
        print(f"[filter_df] state={code!r} → {len(narrowed):,} rows")
        if len(narrowed) > 0:
            result = narrowed
        else:
            print(f"[filter_df] state filter produced 0 rows — skipping")

    # ── City (skip if ZIP is also provided to avoid geographic conflict) ──
    if filters.get("city") and not filters.get("zip"):
        narrowed = result[_col_contains(result["practicecity"], filters["city"])]
        print(f"[filter_df] city={filters['city']!r} → {len(narrowed):,} rows")
        if len(narrowed) > 0:
            result = narrowed
        else:
            print(f"[filter_df] city filter produced 0 rows — skipping")

    # ── ZIP ──
    if filters.get("zip"):
        narrowed = result[result["practicezip9code"].fillna("").str.startswith(str(filters["zip"]))]
        print(f"[filter_df] zip={filters['zip']!r} → {len(narrowed):,} rows")
        if len(narrowed) > 0:
            result = narrowed
        else:
            print(f"[filter_df] zip filter produced 0 rows — skipping")

    # ── Business name ──
    if filters.get("name_contains"):
        mask = (
            _col_contains(result["businessname"], filters["name_contains"]) |
            _col_contains(result["practicename"],  filters["name_contains"])
        )
        narrowed = result[mask]
        print(f"[filter_df] name_contains={filters['name_contains']!r} → {len(narrowed):,} rows")
        if len(narrowed) > 0:
            result = narrowed
        else:
            print(f"[filter_df] name filter produced 0 rows — skipping")

    # ── Supply + Specialty keywords ──────────────────────────────────────────
    # We OR supply_keywords and specialty_keywords together so a supplier that
    # matches EITHER list is included (avoids over-constraining with AND).
    supply_kws  = [k for k in (filters.get("supply_keywords")    or []) if k]
    spec_kws    = [k for k in (filters.get("specialty_keywords") or []) if k]

    if supply_kws or spec_kws:
        combined_mask = pd.Series([False] * len(result), index=result.index)
        matched_any = False

        for kw in supply_kws:
            m = _keyword_mask(result["supplieslist"], kw)
            if m.sum() > 0:
                combined_mask |= m
                matched_any = True

        for kw in spec_kws:
            m = _keyword_mask(result["specialitieslist"], kw)
            if m.sum() > 0:
                combined_mask |= m
                matched_any = True

        if matched_any:
            narrowed = result[combined_mask]
            print(f"[filter_df] supply/specialty OR-combined → {len(narrowed):,} rows")
            result = narrowed
        else:
            print(f"[filter_df] supply/specialty keywords matched nothing — skipping")

    # ── Accepts assignment ──
    if filters.get("accepts_assignment") is not None:
        val = str(filters["accepts_assignment"]).capitalize()
        narrowed = result[result["acceptsassignement"].fillna("").str.capitalize() == val]
        print(f"[filter_df] accepts_assignment={val} → {len(narrowed):,} rows")
        if len(narrowed) > 0:
            result = narrowed
        else:
            print(f"[filter_df] accepts_assignment filter produced 0 rows — skipping")

    # ── CBA contracted ──
    if filters.get("is_contracted_cba") is not None:
        val = str(filters["is_contracted_cba"]).capitalize()
        narrowed = result[result["is_contracted_for_cba"].fillna("").str.capitalize() == val]
        print(f"[filter_df] is_contracted_cba={val} → {len(narrowed):,} rows")
        if len(narrowed) > 0:
            result = narrowed
        else:
            print(f"[filter_df] cba filter produced 0 rows — skipping")

    limit = filters.get("limit") or MAX_ROWS
    limit = max(1, min(int(limit), MAX_ROWS))
    print(f"[filter_df] final result: {len(result):,} rows (returning up to {limit})")
    return result.head(limit)
# ─────────────────────────────────────────────────────────────────────────────


def rows_to_text(result: pd.DataFrame) -> str:
    """Format filtered rows as a compact readable block for the answer prompt."""
    if result.empty:
        return "(no matching records)"

    cols_to_show = [
        "businessname", "practicecity", "practicestate",
        "practiceaddress1", "telephonenumber",
        "specialitieslist", "supplieslist",
        "acceptsassignement", "is_contracted_for_cba",
    ]
    # only include columns that actually exist
    cols_to_show = [c for c in cols_to_show if c in result.columns]
    subset = result[cols_to_show].fillna("")

    lines = [f"Total matching records shown: {len(result)}\n"]
    for i, row in subset.iterrows():
        lines.append(f"--- Record {i+1} ---")
        for col in cols_to_show:
            val = row[col]
            if val:
                lines.append(f"  {col}: {val}")
    return "\n".join(lines)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM response."""
    # try raw parse first
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    # strip markdown fences
    cleaned = re.sub(r"```[a-z]*", "", text).strip().strip("`")
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # grab first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {}


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data     = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    # ── Step 1: Extract filters (non-streaming, temperature=0) ────────────────
    try:
        filter_resp = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system",  "content": FILTER_SYSTEM},
                {"role": "user",    "content": question},
            ],
            options={"temperature": 0, "num_predict": 256},
        )
        filter_text = filter_resp["message"]["content"]
        print(f"[/ask] Gemma raw filter output: {filter_text!r}")
        filters = extract_json(filter_text)
        # Sanitise: Gemma sometimes outputs the string "null" instead of JSON null
        for key in ("state", "city", "zip", "name_contains"):
            if isinstance(filters.get(key), str) and filters[key].strip().lower() in ("null", "none", ""):
                filters[key] = None
        for key in ("supply_keywords", "specialty_keywords"):
            if not isinstance(filters.get(key), list):
                filters[key] = []
        print(f"[/ask] Parsed filters: {filters}")
    except Exception as e:
        filters = {}
        print(f"[filter step error] {e}")

    # ── Step 2: Filter DataFrame ──────────────────────────────────────────────
    matched = filter_df(filters)
    row_count = len(matched)

    # ── Step 3: Stream answer ─────────────────────────────────────────────────
    def generate():
        # Send row count as first SSE event so UI can show it
        yield f"data: __META__{json.dumps({'count': row_count, 'filters': filters})}\n\n"

        if row_count == 0:
            yield "data: No matching suppliers found for your query. Try rephrasing or using broader terms (e.g., just the state, or a general supply keyword).\n\n"
            yield "data: [DONE]\n\n"
            return

        context_text = rows_to_text(matched)
        user_msg = (
            f"Question: {question}\n\n"
            f"Relevant supplier data:\n{context_text}"
        )

        try:
            stream = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": ANSWER_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                stream=True,
                options={"temperature": 0.3, "num_predict": 2048},
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                if token:
                    # Escape newlines for SSE
                    safe = token.replace("\n", "\\n")
                    yield f"data: {safe}\n\n"
        except Exception as e:
            yield f"data: Error generating answer: {e}\n\n"

        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/stats")
def stats():
    return jsonify({
        "total_rows":  len(df),
        "states":      _states,
        "model":       MODEL,
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
