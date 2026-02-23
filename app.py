import os
from pinecone import Pinecone
from dotenv import load_dotenv
from flask import Flask, render_template, request
import google.generativeai as genai


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# -----------------------------
# Initialize Gemini
# -----------------------------
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")


# -----------------------------
# Initialize Flask
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Initialize Pinecone
# -----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)


# -----------------------------
# Detect Namespace
# -----------------------------
def detect_namespace(query):
    q = query.lower()
    if "exporter" in q:
        return "exporters", "exporter"
    elif "importer" in q or "buyer" in q:
        return "importers", "importer"
    elif "news" in q or "risk" in q:
        return "global_news", "news"
    else:
        return "exporters", "exporter"


# -----------------------------
# Retrieve From Pinecone
# -----------------------------
def retrieve(namespace, query, top_k=5):

    response = index.search(
        namespace=namespace,
        query={
            "inputs": {"text": query},
            "top_k": top_k
        }
    )

    return response.get("result", {}).get("hits", [])


# -----------------------------
# Build Context For Gemini
# -----------------------------
def build_context(results):
    context = ""
    for r in results:
        fields = r.get("fields", {})
        context += f"""
ID: {r.get('_id')}
Score: {r.get('_score')}
Details: {fields}
------------------------
"""
    return context


# -----------------------------
# Gemini Formatter
# -----------------------------
def generate_answer(user_query, results, record_type):

    if not results:
        return "⚠️ No relevant records found."

    context = build_context(results)

    prompt = f"""
User Query:
{user_query}

Record Type:
{record_type}

Retrieved Data:
{context}

Instructions:
- Rank results clearly.
- Present in professional trade intelligence style.
- Explain briefly why each match is relevant.
- Keep answer structured and readable.
"""

    response = model.generate_content(prompt)
    return response.text


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    answer = None

    if request.method == "POST":
        user_query = request.form["query"]

        namespace, record_type = detect_namespace(user_query)
        results = retrieve(namespace, user_query)

        answer = generate_answer(user_query, results, record_type)

    return render_template("index.html", answer=answer)


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)