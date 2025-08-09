import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration (should match your converter app) ---
FAISS_DIR = "temp_repo/faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def inspect_vector_store():
    """
    Loads a local FAISS vector store and prints the content of each document.
    """
    if not os.path.exists(FAISS_DIR):
        print(f"Error: Directory '{FAISS_DIR}' not found.")
        print("Please run the converter_app.py first to create the knowledge base.")
        return

    print(f"Loading embeddings with '{MODEL_NAME}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"}
    )

    print(f"Loading FAISS index from '{FAISS_DIR}'...")
    try:
        vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Failed to load FAISS index: {e}")
        return

    # Access the internal document store
    docstore = vectorstore.docstore._dict
    
    if not docstore:
        print("The knowledge base is empty.")
        return

    print(f"\n--- Found {len(docstore)} documents in the knowledge base ---\n")

    for i, doc in enumerate(docstore.values()):
        print(f"---------- Document {i+1} ----------")
        print(doc.page_content)
        print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    inspect_vector_store()
