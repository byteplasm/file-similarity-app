# ----------------------------
# Imports
# ----------------------------
import streamlit as st
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# ----------------------------
# Functions
# ----------------------------
def get_text(file):
    """Extract text from uploaded PDF or TXT file."""
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return " ".join(text.split())
    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8", errors="ignore")
        return " ".join(text.split())
    return ""

def similarity(vec1, vec2):
    """Cosine similarity between two vectors"""
    if vec1 is None or vec2 is None:
        return 0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📁 File Similarity & Duplicate Manager")

# Upload files
uploaded_files = st.file_uploader(
    "Upload files to analyze",
    accept_multiple_files=True
)

# Similarity threshold
threshold = st.slider(
    "Similarity threshold (loose → strict)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Lower = more files considered similar, higher = only very similar files"
)
st.write(f"Current threshold: {threshold:.2f}")

# Initialize session state for decisions
if "decisions" not in st.session_state:
    st.session_state.decisions = {}

if uploaded_files:
    texts = {}
    embeddings = {}

    # Extract text & embeddings
    for file in uploaded_files:
        text = get_text(file)
        texts[file.name] = text
        embeddings[file.name] = model.encode(text)

    st.subheader("Compare Files")

    # Button triggers comparison
    if st.button("Compare Files"):
        names = list(embeddings.keys())
        any_similar = False

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name1 = names[i]
                name2 = names[j]

                vec1 = embeddings[name1]
                vec2 = embeddings[name2]

                score = similarity(vec1, vec2)

                if score >= threshold:
                    any_similar = True
                    st.write(f"Similarity: {score:.2f}")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"📄 {name1}")
                        st.text_area("Preview", texts[name1], height=150)
                        if st.button(f"Keep {name1}", key=f"{name1}-{name2}-A"):
                            st.session_state.decisions[(name1, name2)] = name1

                    with col2:
                        st.write(f"📄 {name2}")
                        st.text_area("Preview", texts[name2], height=150)
                        if st.button(f"Keep {name2}", key=f"{name1}-{name2}-B"):
                            st.session_state.decisions[(name1, name2)] = name2

                    # Ignore button
                    if st.button(f"Ignore", key=f"{name1}-{name2}-Ignore"):
                        st.session_state.decisions[(name1, name2)] = "Ignore"

                    st.write("---")

        if not any_similar:
            st.write("No similar files found above the threshold.")

    # Show decisions
    if st.session_state.decisions:
        st.subheader("✅ Decisions so far")
        for pair, decision in st.session_state.decisions.items():
            st.write(f"{pair[0]} ⇄ {pair[1]} → {decision}")