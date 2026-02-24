import os
from pinecone import Pinecone
from flask import Flask, render_template, request
import google.generativeai as genai
import markdown
import threading

# -----------------------------
# Load environment variables
# -----------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX")

# -----------------------------
# Gemini Key Rotation (Round Robin)
# Set GEMINI_API_KEYS as comma-separated keys in HF Secrets
# e.g.  key1,key2,key3
# -----------------------------
GEMINI_API_KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]

if not GEMINI_API_KEYS:
    raise Exception("❌ No GEMINI_API_KEYS found. Add them in HF Spaces Secrets.")

key_index = 0
key_lock  = threading.Lock()

def get_next_model():
    """Returns a Gemini model configured with the next API key (round robin)."""
    global key_index
    with key_lock:
        api_key    = GEMINI_API_KEYS[key_index]
        key_index  = (key_index + 1) % len(GEMINI_API_KEYS)
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")

# -----------------------------
# Initialize Flask
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Initialize Pinecone
# -----------------------------
pc    = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# -----------------------------
# All Available Namespaces
# -----------------------------
ALL_NAMESPACES = {
    "exporters":  "exporter",
    "importers":  "importer",
    "globalnews": "news"
}

# -----------------------------
# Smart Namespace Detector
# -----------------------------
def detect_namespaces(query):
    q = query.lower()

    keyword_map = {
        "exporters":  ["exporter", "export", "supplier", "seller", "manufacturer"],
        "importers":  ["importer", "import", "buyer", "purchaser", "procurement"],
        "globalnews": ["news", "risk", "market", "global", "alert", "trend", "update",
                       "affect", "affected", "impact", "disruption", "sanction",
                       "tariff", "policy", "regulation", "report", "forecast", "outlook"]
    }

    matched = [ns for ns, keywords in keyword_map.items() if any(kw in q for kw in keywords)]

    # Nothing matched → broad query → search everything
    if not matched:
        matched = list(ALL_NAMESPACES.keys())

    return matched


# -----------------------------
# Retrieve From ONE Namespace
# -----------------------------
def retrieve_from_namespace(namespace, query, top_k=15):
    try:
        response = index.search(
            namespace=namespace,
            query={"inputs": {"text": query}, "top_k": top_k}
        )
        hits = response.get("result", {}).get("hits", [])
        for hit in hits:
            hit["_namespace"]    = namespace
            hit["_record_type"]  = ALL_NAMESPACES[namespace]
        return hits
    except Exception as e:
        print(f"[ERROR] Namespace '{namespace}' failed: {e}")
        return []


# -----------------------------
# Multi-Namespace Search + Merge
# -----------------------------
def retrieve(query, top_k_per_ns=15):
    namespaces = detect_namespaces(query)
    all_hits   = []

    for ns in namespaces:
        all_hits.extend(retrieve_from_namespace(ns, query, top_k=top_k_per_ns))

    all_hits.sort(key=lambda x: x.get("_score", 0), reverse=True)
    return all_hits, namespaces


# -----------------------------
# Build Context For Gemini
# -----------------------------
def build_context(results):
    context = ""
    for r in results:
        fields   = r.get("fields", {})
        context += f"""
ID: {r.get('_id')}
Source: {r.get('_namespace', 'unknown')} ({r.get('_record_type', '')})
Score: {r.get('_score')}
Details: {fields}
------------------------
"""
    return context


# -----------------------------
# Gemini Answer Generator (with Round Robin + 429 fallback)
# -----------------------------
def generate_answer(user_query, results, searched_namespaces):

    if not results:
        return "⚠️ **Not able to answer this query based on available trade data.**"

    context  = build_context(results)
    ns_label = ", ".join(searched_namespaces)

    prompt = f"""
You are TIPE AI – Trade Intent Prediction Engine.

User Query:
{user_query}

Namespaces Searched:
{ns_label}

Retrieved Trade Records (merged & ranked by relevance score):
{context}

IMPORTANT RULES:

1. ALWAYS try to generate a report using the retrieved data, even if the match is partial.
   Only refuse if retrieved data has ZERO connection to the query. Do NOT refuse just because
   an industry keyword like "engineering" or "pharma" is not an exact field match — use context clues.

2. Generate a structured professional trade intelligence report.

3. If user specified a count like "3" or "3-4", show only that many ranked results and that many strategic insights. Otherwise default to 5 each.

4. Each result must clearly show which data source it came from (Exporter / Importer / News).

STRICT FORMAT:

# Trade Intelligence Report

## Executive Summary
(2-3 lines summarizing findings across all searched sources)

## Ranked Results

For each result:
**Rank X — [Source Type]**
- **Entity ID:**
- **Location:**
- **Industry:**
- **Revenue:**
- **Intent Score:**
- **Risk Indicators:**
- **Why Relevant:**

## Strategic Insights
- Insight 1
- Insight 2
- Insight 3

## Data Sources Searched
{ns_label}

DO NOT mention raw JSON or technical metadata.
Keep formatting clean with markdown-style bold headings.
"""

    # Try each key in round robin — skip on 429, raise on other errors
    last_error = None
    for _ in range(len(GEMINI_API_KEYS)):
        try:
            model    = get_next_model()
            response = model.generate_content(prompt)
            output   = response.text.strip()

            refusal_phrases = [
                "not able to answer based on available trade intelligence",
                "not able to answer this query"
            ]
            is_refusal = any(p in output.lower() for p in refusal_phrases)
            if is_refusal and len(output) < 200:
                return "⚠️ **Not able to answer this query based on available trade data.**"

            return output

        except Exception as e:
            last_error = e
            if "429" in str(e):
                print(f"[WARN] 429 quota hit, rotating to next key...")
                continue  # try next key
            else:
                return f"⚠️ **AI processing error:** `{str(e)}`"

    return f"⚠️ **All {len(GEMINI_API_KEYS)} Gemini API keys exhausted (quota reached). Try later.** Last error: `{str(last_error)}`"


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    answer = None

    if request.method == "POST":
        user_query = request.form["query"]

        results, searched_namespaces = retrieve(user_query)
        raw_answer = generate_answer(user_query, results, searched_namespaces)

        answer = markdown.markdown(raw_answer)

    return render_template("index.html", answer=answer)


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
