import streamlit as st
import os
import hashlib
from PyPDF2 import PdfReader

def get_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    return ""
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# FILE UTILITIES
# ----------------------------

def get_file_hash(file_path):
    """Exact duplicate detection"""
    hasher = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    except:
        return None


def extract_text(file_path):
    """Extract text from txt and pdf files"""
    try:
        if file_path.endswith(".pdf"):
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text

        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    except:
        return ""

    return ""


def compare_files(file1, file2):
    """Semantic similarity using TF-IDF"""
    text1 = extract_text(file1)
    text2 = extract_text(file2)

    if not text1 or not text2:
        return 0

    try:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([text1, text2])
        score = cosine_similarity(vectors[0], vectors[1])[0][0]
        return score
    except:
        return 0


# ----------------------------
# SCAN FUNCTIONS
# ----------------------------

def scan_folder(folder_path):
    files = []
    hashes = {}
    duplicates = []

    for root, _, filenames in os.walk(folder_path):
        for name in filenames:
            path = os.path.join(root, name)
            files.append(path)

            file_hash = get_file_hash(path)
            if file_hash:
                if file_hash in hashes:
                    duplicates.append((hashes[file_hash], path))
                else:
                    hashes[file_hash] = path

    return files, duplicates


def find_similar_files(files, threshold=0.75):
    similar_pairs = []

    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            score = compare_files(files[i], files[j])

            if score >= threshold:
                similar_pairs.append((files[i], files[j], score))

    return similar_pairs


# ----------------------------
# STREAMLIT UI
# ----------------------------

st.title("📁 File Similarity & Duplicate Detector")

uploaded_files = st.file_uploader(
    "Upload files to analyze",
    accept_multiple_files=True
)
if uploaded_files:
    st.write("Files uploaded:")
    for file in uploaded_files:
        st.write("📄", file.name)

threshold = st.slider("Similarity threshold", 0.5, 1.0, 0.75)

if st.button("Scan Folder"):

    if not os.path.exists(folder):
        st.error("Folder does not exist!")
    else:
        with st.spinner("Scanning files..."):

            files, duplicates = scan_folder(folder)
            similar = find_similar_files(files, threshold)

        # ---------------- DUPLICATES ----------------
        st.subheader("🔴 Exact Duplicates")

        if duplicates:
            for a, b in duplicates:
                st.write(f"📄 {a}")
                st.write(f"📄 {b}")
                st.write("---")
        else:
            st.write("No exact duplicates found.")

        # ---------------- SIMILAR FILES ----------------
        st.subheader("🟡 Similar Files")

        if similar:
            for a, b, score in similar:
                st.write(f"📄 {a}")
                st.write(f"📄 {b}")
                st.write(f"Similarity: {score:.2f}")
                st.write("---")
        else:
            st.write("No similar files found.")