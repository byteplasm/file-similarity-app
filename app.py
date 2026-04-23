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
    """
    Extract text from uploaded PDF or TXT file.
    """
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    return ""


def similarity(vec1, vec2):
    """Cosine similarity between two vectors"""
    if vec1 is None or vec2 is None:
        return 0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# ----------------------------
# Load AI model
# ----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()


# ----------------------------
# Streamlit UI
# ----------------------------

st.title("📁 File Similarity & Duplicate Detector")

# Upload multiple files
uploaded_files = st.file_uploader(
    "Upload files to analyze",
    accept_multiple_files=True
)

# Slider for similarity threshold
threshold = st.slider("Similarity threshold", 0.5, 1.0, 0.75)

if uploaded_files:
    st.subheader("Uploaded Files")
    for file in uploaded_files:
        st.write("📄", file.name)

    texts = {}
    embeddings = {}

    # Extract text and embeddings
    for file in uploaded_files:
        text = get_text(file)
        texts[file.name] = text
        embeddings[file.name] = model.encode(text)

    st.subheader("Compare Files")

    # Button triggers similarity comparison
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
                    st.write(f"📄 {name1}")
                    st.write(f"📄 {name2}")
                    st.write(f"Similarity: {score:.2f}")
                    st.write("---")

        if not any_similar:
            st.write("No similar files found above the threshold.")