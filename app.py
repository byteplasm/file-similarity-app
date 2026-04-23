import streamlit as st
import os
import shutil
import hashlib
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# ----------------------------
# Functions
# ----------------------------
def get_file_text(file_path):
    """Extract text from supported files; fallback None for others."""
    try:
        if file_path.endswith(".pdf"):
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return " ".join(text.split())
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return " ".join(f.read().split())
        else:
            return None
    except:
        return None

def file_hash(file_path):
    """Compute MD5 hash for exact duplicate detection"""
    hasher = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except:
        return None

def similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def group_similar_files(file_paths, embeddings, threshold=0.5):
    groups = []
    visited = set()
    for i in range(len(file_paths)):
        if file_paths[i] in visited:
            continue
        group = [file_paths[i]]
        visited.add(file_paths[i])
        for j in range(i+1, len(file_paths)):
            if file_paths[j] in visited:
                continue
            score = similarity(embeddings[file_paths[i]], embeddings[file_paths[j]])
            if score >= threshold:
                group.append(file_paths[j])
                visited.add(file_paths[j])
        if len(group) > 1:
            groups.append(group)
    return groups

def auto_rename(file_path, folder_path):
    base, ext = os.path.splitext(os.path.basename(file_path))
    counter = 1
    new_name = f"{base}_duplicate{counter}{ext}"
    new_path = os.path.join(folder_path, new_name)
    while os.path.exists(new_path):
        counter += 1
        new_name = f"{base}_duplicate{counter}{ext}"
        new_path = os.path.join(folder_path, new_name)
    shutil.move(file_path, new_path)
    return new_path

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
st.title("📁 Universal File Duplicate & Similarity Manager")

folder_path = st.text_input("Enter folder path to scan:")

threshold = st.slider("Similarity threshold (text files only)", 0.0, 1.0, 0.5, 0.01)

cleanup_mode = st.radio(
    "Duplicate handling mode",
    options=[
        "Move duplicates to _duplicates folder",
        "Auto-rename duplicates in place"
    ]
)

if folder_path and os.path.exists(folder_path):
    st.write(f"Scanning folder: `{folder_path}`")
    
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    texts = {}
    embeddings = {}
    hashes = {}
    exact_duplicates = []
    
    # Process files
    for path in file_paths:
        # Hash for exact duplicates
        h = file_hash(path)
        if h:
            if h in hashes:
                exact_duplicates.append((hashes[h], path))
            else:
                hashes[h] = path
        
        # Text extraction
        text = get_file_text(path)
        if text:
            texts[path] = text
            embeddings[path] = model.encode(text)
    
    # Show exact duplicates
    st.subheader("🔴 Exact Duplicates (hash match)")
    if exact_duplicates:
        for a, b in exact_duplicates:
            st.write(f"📄 {a}")
            st.write(f"📄 {b}")
            st.write("---")
    else:
        st.write("No exact duplicates found.")
    
    # Group similar text files
    if embeddings:
        st.subheader("🟡 Similar Text Files (semantic similarity)")
        groups = group_similar_files(list(embeddings.keys()), embeddings, threshold)
        
        if not groups:
            st.write("No similar text file groups found above threshold.")
        else:
            decisions = {}
            for idx, group in enumerate(groups):
                st.write(f"### Group {idx+1}")
                keep_file = st.selectbox(
                    "Select file to keep (others handled automatically)",
                    options=group,
                    key=f"keep-{idx}"
                )
                decisions[idx] = {"keep": keep_file, "group": group}
                for jdx, path in enumerate(group):
                    with st.expander(f"Preview: {os.path.basename(path)}"):
                        st.text_area("Preview", texts[path], height=150, key=f"preview-{idx}-{jdx}")
            
            # Cleanup button
            if st.button("Run Automatic Cleanup"):
                moved_files = []
                
                # Handle exact duplicates
                for a, b in exact_duplicates:
                    for f in [a, b]:
                        if cleanup_mode == "Move duplicates to _duplicates folder":
                            archive_folder = os.path.join(folder_path, "_duplicates")
                            os.makedirs(archive_folder, exist_ok=True)
                            dst = os.path.join(archive_folder, os.path.basename(f))
                            if os.path.exists(f):
                                shutil.move(f, dst)
                                moved_files.append({"Moved": f, "To": dst})
                        else:  # Auto-rename
                            new_path = auto_rename(f, folder_path)
                            moved_files.append({"Renamed": f, "To": new_path})
                
                # Handle semantic duplicates
                for decision in decisions.values():
                    keep = decision["keep"]
                    for f in decision["group"]:
                        if f != keep:
                            if cleanup_mode == "Move duplicates to _duplicates folder":
                                archive_folder = os.path.join(folder_path, "_duplicates")
                                os.makedirs(archive_folder, exist_ok=True)
                                dst = os.path.join(archive_folder, os.path.basename(f))
                                if os.path.exists(f):
                                    shutil.move(f, dst)
                                    moved_files.append({"Moved": f, "To": dst})
                            else:
                                new_path = auto_rename(f, folder_path)
                                moved_files.append({"Renamed": f, "To": new_path})
                
                if moved_files:
                    st.success(f"Processed {len(moved_files)} duplicate files!")
                    st.dataframe(pd.DataFrame(moved_files))
                else:
                    st.info("No files needed processing.")