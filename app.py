import streamlit as st
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import os

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

def group_similar_files(names, embeddings, threshold=0.5):
    """Group files that are similar to each other"""
    groups = []
    visited = set()

    for i in range(len(names)):
        if names[i] in visited:
            continue
        group = [names[i]]
        visited.add(names[i])
        for j in range(i + 1, len(names)):
            if names[j] in visited:
                continue
            score = similarity(embeddings[names[i]], embeddings[names[j]])
            if score >= threshold:
                group.append(names[j])
                visited.add(names[j])
        if len(group) > 1:
            groups.append(group)
    return groups

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
st.title("📁 File Similarity & Bulk Duplicate Manager with Export")

uploaded_files = st.file_uploader(
    "Upload files to analyze",
    accept_multiple_files=True
)

threshold = st.slider(
    "Similarity threshold (loose → strict)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Lower = more files considered similar, higher = only very similar files"
)
st.write(f"Current threshold: {threshold:.2f}")

# Session state for bulk decisions
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

    st.subheader("Uploaded Files Preview")
    for file in uploaded_files:
        with st.expander(f"📄 {file.name} (click to show/hide preview)"):
            st.text_area(f"Preview of {file.name}", texts[file.name], height=150)

    st.subheader("Bulk Similar File Groups")
    if st.button("Find Similar Groups"):
        names = list(embeddings.keys())
        groups = group_similar_files(names, embeddings, threshold)

        if not groups:
            st.write("No similar file groups found above the threshold.")
        else:
            for idx, group in enumerate(groups):
                st.write(f"### Group {idx+1} (similar files)")
                selected_keep = st.radio(
                    f"Choose file to keep for Group {idx+1}",
                    options=group + ["Keep All", "Ignore"],
                    key=f"group-{idx}"
                )
                for name in group:
                    with st.expander(f"📄 {name} (click to show/hide preview)"):
                        st.text_area("Preview", texts[name], height=150)
                st.session_state.decisions[f"group-{idx}"] = selected_keep

    # -----------------------
    # Show decisions
    # -----------------------
    if st.session_state.decisions:
        st.subheader("✅ Bulk Decisions")
        for group_id, decision in st.session_state.decisions.items():
            st.write(f"{group_id} → {decision}")

        # Export decisions
        st.subheader("📥 Export Decisions")
        if st.button("Download CSV of Decisions"):
            df = pd.DataFrame([
                {"Group": group_id, "Decision": decision}
                for group_id, decision in st.session_state.decisions.items()
            ])
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="file_decisions.csv",
                mime="text/csv"
            )

        # Optional: Local rename/move (only works if files are on local disk)
        # Example placeholder (uncomment if running locally):
        # for group_id, decision in st.session_state.decisions.items():
        #     if decision not in ["Keep All", "Ignore"]:
        #         # implement renaming/moving logic here