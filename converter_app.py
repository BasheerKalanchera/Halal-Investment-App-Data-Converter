import streamlit as st
import os
import requests
import pickle
import hashlib
from git import Repo
from docx import Document as DocxDocument

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# --- File Paths and URLs ---
REPO_URL = "https://github.com/BasheerKalanchera/Halal-Investment-App-Data-Converter.git"
FAISS_FILE = "knowledge_base.faiss"
DOCS_FILE = "knowledge_base.pkl"
LOCAL_CLONE_PATH = "./temp_repo"

# --- GitHub Authentication ---
try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
except KeyError:
    st.error("GitHub Personal Access Token not found. Please set it in Streamlit secrets or your .env file.")
    st.stop()

# --- Embedding model (same as RAG app) ---
@st.cache_resource
def load_embedding_model():
    # The converter app will still load the hkunlp/instructor-large embedding model
    # to ensure consistency with the FAISS index[cite: 567].
    return HuggingFaceEmbeddings(
        model_name="hkunlp/instructor-large",
        model_kwargs={"device": "cpu"}
    )

# --- Document Parsing Function ---
def parse_docx_for_qas(file_path):
    """Parses a .docx file and extracts Q&A pairs."""
    try:
        doc = DocxDocument(file_path)
        qa_pairs = []
        q_text = None
        a_text = None
        for para in doc.paragraphs:
            if para.text.strip().startswith("Question"):
                if q_text and a_text:
                    qa_pairs.append(Document(page_content=f"Question\n{q_text}\nAnswer\n{a_text}"))
                q_text = para.text.replace("Question", "").strip()
                a_text = None
            elif para.text.strip().startswith("Answer"):
                a_text = para.text.replace("Answer", "").strip()
            elif q_text and a_text is None:
                q_text += " " + para.text.strip()
            elif a_text is not None:
                a_text += " " + para.text.strip()
        if q_text and a_text:
            qa_pairs.append(Document(page_content=f"Question\n{q_text}\nAnswer\n{a_text}"))
        return qa_pairs
    except Exception as e:
        st.error(f"Error parsing .docx file: {e}")
        return []

# --- Main app logic ---
def run_converter_app():
    st.title("📚 Halal Gold Knowledge Base Converter")
    st.info("Upload a .docx file to update the app's knowledge base on GitHub.")

    uploaded_file = st.file_uploader("Choose a .docx file", type=["docx"])

    if uploaded_file and st.button("Update Knowledge Base"):
        with st.spinner("Processing new Q&As..."):
            # Save uploaded file to a temporary location
            temp_docx_path = f"temp_{uploaded_file.name}"
            with open(temp_docx_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 1. Load existing FAISS index and documents from GitHub[cite: 556].
            # This requires cloning the repository or fetching the files.
            # Using requests to fetch raw files from GitHub is simpler for this example.
            st.write("Fetching existing knowledge base from GitHub...")
            try:
                # The app will download and load the pre-built .faiss and .pkl files from GitHub.
                faiss_url = f"https://raw.githubusercontent.com/BasheerKalanchera/Halal-Gold-Investment-Assistant/main/{FAISS_FILE}"
                docs_url = f"https://raw.githubusercontent.com/BasheerKalanchera/Halal-Gold-Investment-Assistant/main/{DOCS_FILE}"
                faiss_data = requests.get(faiss_url).content
                docs_data = requests.get(docs_url).content
                with open(FAISS_FILE, "wb") as f: f.write(faiss_data)
                with open(DOCS_FILE, "wb") as f: f.write(docs_data)

                st.write("Loading FAISS index and documents...")
                embeddings = load_embedding_model()
                vectorstore = FAISS.load_local(FAISS_FILE, embeddings, allow_dangerous_deserialization=True)
                with open(DOCS_FILE, "rb") as f:
                    existing_docs = pickle.load(f)
            except Exception as e:
                st.warning(f"Could not load existing knowledge base from GitHub. Starting with an empty index. Error: {e}")
                embeddings = load_embedding_model()
                vectorstore = FAISS.from_texts([" "], embeddings)
                existing_docs = []

            # 2. Parse new documents.
            new_docs = parse_docx_for_qas(temp_docx_path)
            st.success(f"Parsed {len(new_docs)} new Q&A pairs from the .docx file.")
            os.remove(temp_docx_path)

            # 3. Combine and rebuild index[cite: 561].
            all_docs = existing_docs + new_docs
            # Rebuilds the entire FAISS index from the combined, updated set of Q&A documents[cite: 561].
            st.write("Rebuilding FAISS index...")
            vectorstore = FAISS.from_documents(all_docs, embeddings)

            # 4. Save new FAISS and documents files.
            st.write("Saving new index files locally...")
            vectorstore.save_local(FAISS_FILE)
            with open(DOCS_FILE, "wb") as f:
                pickle.dump(all_docs, f)

            # 5. Push files to GitHub.
            st.write("Pushing updated knowledge base to GitHub...")
            try:
                repo_path = LOCAL_CLONE_PATH
                if os.path.exists(repo_path):
                    repo = Repo(repo_path)
                    repo.git.pull()
                else:
                    repo_with_token = REPO_URL.replace("https://", f"https://oauth2:{GITHUB_TOKEN}@")
                    repo = Repo.clone_from(repo_with_token, repo_path)

                # Copy new files into the repository
                os.rename(FAISS_FILE, os.path.join(repo_path, FAISS_FILE))
                os.rename(DOCS_FILE, os.path.join(repo_path, DOCS_FILE))

                repo.index.add([FAISS_FILE, DOCS_FILE])
                repo.index.commit("Automated knowledge base update via converter app.")
                repo.git.push("origin", "main")
                st.success("Knowledge base successfully updated and pushed to GitHub!")
                # Logging: Logs all actions[cite: 563].
                st.info("Log: Automated update on " + str(datetime.now()))
            except Exception as e:
                st.error(f"Failed to push to GitHub: {e}")

if __name__ == "__main__":
    run_converter_app()