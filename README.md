# cms-query
Create a Flask web application for querying a medical equipment suppliers CSV database using Ollama with local LLM (Gemma 3:4b model).

REQUIREMENTS:

1. DATA & SETUP:
   - Load "Medical-Equipment-Suppliers.csv" at startup into pandas DataFrame
   - Columns: provider_id, acceptsassignement, participationbegindate, businessname, practicename, practiceaddress1, practicecity, practicestate (2-letter codes), practicezip9code, telephonenumber, specialitieslist (pipe-separated), providertypelist (pipe-separated), supplieslist (pipe-separated), latitude, longitude, is_contracted_for_cba
   - Extract unique states, top 40 supplies, and top 30 specialties as hints for prompts

2. TWO-PHASE LLM APPROACH:
   - PHASE 1 (Filter Extraction): Use Ollama non-streaming with temperature=0 to extract structured JSON filters from natural language questions. Return: {"state": "2-letter code or null", "city": "UPPERCASE or null", "supply_keywords": [], "specialty_keywords": [], "name_contains": "string or null", "accepts_assignment": true/false/null, "is_contracted_cba": true/false/null, "zip": "prefix or null", "limit": 30}
   - PHASE 2 (Answer Generation): Use Ollama streaming with temperature=0.3 to generate natural language answers based on filtered data
   - Use Server-Sent Events (SSE) to stream responses to frontend

3. FILTER EXTRACTION PROMPT RULES:
   - State MUST be 2-letter uppercase code (FL not Florida, TX not Texas)
   - City MUST be UPPERCASE and ONLY set if explicitly mentioned (no guessing)
   - ZIP only set if explicitly mentioned
   - Supply keywords must be SHORT (1-3 words): "Oxygen", "Wheelchair", "CPAP" - NOT full phrases like "Oxygen Equipment and/or Supplies"
   - Specialty keywords also short: "Pharmacy", "Orthotic", "Prosthetic"
   - Only populate fields explicitly stated by user - use null/[] for everything else
   - Never invent or guess location details
   - Include list of known state codes, common supplies, and specialties in system prompt

4. SMART FILTERING WITH GRACEFUL DEGRADATION:
   - Apply filters progressively: state → city → zip → name → supplies/specialties → boolean flags
   - If ANY filter produces 0 results, SKIP that filter and continue with others
   - Use OR logic for supply_keywords and specialty_keywords combined (match EITHER list)
   - Handle state name to 2-letter code conversion with dictionary mapping all 50 states + DC
   - City filter: skip if ZIP also provided to avoid geographic conflicts
   - Keyword matching: split multi-word keywords and require ALL words to appear (AND within keyword, OR across keywords)
   - Limit results to maximum 30 rows
   - Print detailed debug logs for each filter step showing row counts

5. ANSWER GENERATION:
   - Format filtered results as readable text context (show: businessname, practicecity, practicestate, practiceaddress1, telephonenumber, specialitieslist, supplieslist, acceptsassignement, is_contracted_for_cba)
   - Combine user question + formatted data context and send to Ollama
   - Stream response token-by-token via SSE
   - Send metadata (row count, filters used) as first SSE event with "__META__" prefix
   - Handle empty results with helpful message suggesting broader terms

6. FRONTEND (templates/index.html):
   - Modern dark theme UI (#0f1117 background, #1a1d27 panels)
   - Header with gradient logo, title "Medical Equipment Supplier Q&A", live pill showing row count
   - Empty state with icon, examples like "Find oxygen suppliers in Florida", "CPAP equipment in Miami"
   - Chat interface with user/bot message bubbles and avatars
   - User input at bottom with send button
   - Use Server-Sent Events (EventSource) to receive streaming responses
   - Parse "\n" escape sequences in SSE data back to newlines
   - Support example buttons that auto-fill questions
   - Show typing indicator while AI responds
   - Use marked.js to render markdown in bot responses

7. FLASK ROUTES:
   - GET / → serve index.html
   - POST /ask → accept {"question": "..."}, return SSE stream with answer
   - GET /stats → return {"total_rows": X, "states": [...], "model": "gemma3:4b"}

8. ERROR HANDLING:
   - Extract JSON from LLM responses with fallback regex for markdown fences
   - Sanitize "null" strings to actual null values
   - Handle missing columns gracefully
   - Catch Ollama exceptions and send error messages via SSE

9. CONFIGURATION:
   - Model: "gemma3:4b"
   - Max rows: 30
   - Filter prompt: temperature=0, num_predict=256
   - Answer prompt: temperature=0.3, num_predict=2048
   - Flask: debug=False, host="0.0.0.0", port=5000, threaded=True

10. FILE STRUCTURE:
   - app.py (main Flask application)
   - templates/index.html (frontend UI)
   - requirements.txt (flask, pandas, ollama)
   - Medical-Equipment-Suppliers.csv (data file)

CRITICAL IMPLEMENTATION DETAILS:
- Use ollama.chat() not ollama.generate()
- Set response headers: Cache-Control: no-cache, X-Accel-Buffering: no
- Use stream_with_context() from Flask for SSE
- Escape newlines in tokens as "\n" before sending via SSE
- Send "[DONE]" as final SSE event
- Function _keyword_mask should skip words <= 2 chars to avoid noise
- Use fillna("") before string operations to handle NaN values
- Normalize all column names to lowercase at startup
