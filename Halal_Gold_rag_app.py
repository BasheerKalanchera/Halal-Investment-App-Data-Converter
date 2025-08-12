import streamlit as st
import os
import hashlib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pickle
import requests
import git

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.docstore.document import Document
from dotenv import load_dotenv

# --- Load .env for local development ---
load_dotenv()
import streamlit.components.v1 as components
import time

def scroll_to_bottom():
    """A small JavaScript component that scrolls the chat container to the bottom."""
    js = f"""
    <script>
        function scroll(dummy_var_to_force_re_execution) {{
            var chat_container = parent.document.querySelector('div[style*="height: 400px"]');
            if (chat_container) {{
                chat_container.scrollTop = chat_container.scrollHeight;
            }}
        }}
        scroll({time.time()});
    </script>
    """
    components.html(js, height=0, scrolling=False)

# --- GCP Service Account Authentication ---
GCP_SA = None
try:
    GCP_SA = st.secrets["gcp_service_account"]
except Exception:
    # Fallback to .env for local development
    try:
        gcp_sa_details = {
            "type": os.getenv("GCP_TYPE"),
            "project_id": os.getenv("GCP_PROJECT_ID"),
            "private_key_id": os.getenv("GCP_PRIVATE_KEY_ID"),
            "private_key": os.getenv("GCP_PRIVATE_KEY").replace("\\n", "\n"),
            "client_email": os.getenv("GCP_CLIENT_EMAIL"),
            "client_id": os.getenv("GCP_CLIENT_ID"),
            "auth_uri": os.getenv("GCP_AUTH_URI"),
            "token_uri": os.getenv("GCP_TOKEN_URI"),
            "auth_provider_x509_cert_url": os.getenv("GCP_AUTH_PROVIDER_X509_CERT_URL"),
            "client_x509_cert_url": os.getenv("GCP_CLIENT_X509_CERT_URL"),
        }
        if all(gcp_sa_details.values()):
            GCP_SA = gcp_sa_details
    except Exception as e:
        st.error(f"Error loading GCP service account from .env: {e}")
        st.stop()

if not GCP_SA:
    st.error("GCP service account details not found. Please set them in Streamlit secrets or your .env file.")
    st.stop()


# --- Constants ---
FAISS_DIR = "faiss_index"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/BasheerKalanchera/Halal-Investment-App-Data-Converter/main"

# Ensure your read-only PAT is also loaded from .env
GITHUB_READ_TOKEN = os.getenv("GITHUB_READ_TOKEN")


# --- Logging unanswered questions to Google Sheets ---
def log_unanswered_to_google_sheets(user_query, user_id="Anonymous"):
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(GCP_SA, scope)
        client = gspread.authorize(creds)
        sheet = client.open("HalalGold_UnansweredLogs").sheet1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([timestamp, user_id, user_query])
    except Exception as e:
        st.warning(f"⚠️ Logging to Google Sheets failed: {e}")

# --- Embedding model ---
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

# --- Retriever setup (FAST STARTUP VERSION) ---
@st.cache_resource(show_spinner=False)
def create_retriever_from_github():
    with st.spinner("Loading up your Aisar - Halal Gold investment assistant... Please wait."):
        _embeddings = load_embedding_model()

        #st.write("Downloading knowledge base from GitHub...")
        try:
            headers = {"Authorization": f"token {GITHUB_READ_TOKEN}"}
            faiss_url = f"{GITHUB_RAW_URL}/{FAISS_DIR}/index.faiss"
            pkl_url = f"{GITHUB_RAW_URL}/{FAISS_DIR}/index.pkl"

            faiss_data = requests.get(faiss_url, headers=headers).content
            pkl_data = requests.get(pkl_url, headers=headers).content

            os.makedirs(FAISS_DIR, exist_ok=True)
            with open(os.path.join(FAISS_DIR, "index.faiss"), "wb") as f:
                f.write(faiss_data)
            with open(os.path.join(FAISS_DIR, "index.pkl"), "wb") as f:
                f.write(pkl_data)

            #st.write("Loading FAISS index...")
            vectorstore = FAISS.load_local(FAISS_DIR, _embeddings, allow_dangerous_deserialization=True)

            #st.success("✅ Knowledge base loaded successfully!")
        
            return vectorstore.as_retriever(search_kwargs={"k": 10})

        except Exception as e:
            st.error(f"❌ Failed to download or load knowledge base from GitHub: {e}")
            st.stop()



# --- Load LLM and retriever ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
retriever = create_retriever_from_github()


# --- Prompt template ---
prompt_template = """
You are a helpful AI assistant specialized in Gold ETFs. Your task is to answer the user's question based strictly on the provided context.
Read the context carefully and synthesize the information that directly answers the question.
If the context provides a clear list of steps, methods, or types that are relevant to the user's question, it is helpful to present your answer as a numbered or bulleted list.
If the context does not contain information to answer the question, you must respond with: "I don't have enough information in the provided documents to answer this question." Do not use any external knowledge.

Context:
{context}

Question: {question}

Answer:
"""

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": PromptTemplate(template=prompt_template, input_variables=["context", "question"])}
)

# --- Streamlit UI ---
st.set_page_config(page_title="Halal Gold Investment Assistant", layout="wide")

st.title("💰 Aisar - Halal Gold Investment Assistant")

st.warning(
    """
**Disclaimer:** This tool is not intended as investment or Shariah advice. Users should not rely on its responses for financial decisions and are advised to consult with a registered financial advisor or qualified Shariah scholar.
"""
)

# Optional user ID in sidebar for a cleaner look
with st.sidebar:
    st.header("User Details")
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = ""
    st.session_state["user_id"] = st.text_input(
        "Enter your name or email (optional):", value=st.session_state["user_id"]
    )


# --- Chat History Implementation ---

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create a container for the chat history
chat_container = st.container(height=400, border=False)

# Display prior chat messages from session state
for message in st.session_state.messages:
    with chat_container.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if user_query := st.chat_input("Enter your question about Halal ways to invest in Gold here:"):
    # Add user message to state and immediately display it
    st.session_state.messages.append({"role": "user", "content": user_query})
    with chat_container.chat_message("user"):
        st.markdown(user_query)

    # Use a placeholder for the assistant's response
    with chat_container.chat_message("assistant"):
        # 1. Create a single placeholder
        placeholder = st.empty()

        try:
            # 2. Put a temporary message in the placeholder
            placeholder.markdown("Searching and generating answer...")
            
            # 3. Get the actual answer
            result = qa_chain.invoke({"query": user_query})
            answer = result["result"]
            
            # 4. Replace the temporary message with the final answer
            placeholder.markdown(answer)

            # Log to Google Sheets if the answer is a fallback
            fallback_message = "I don't have enough information in the provided documents to answer this question."
            if fallback_message in answer:
                log_unanswered_to_google_sheets(
                    user_query.strip(), st.session_state["user_id"] or "Anonymous"
                )
        except Exception as e:
            answer = f"An error occurred: {e}"
            # Or replace with an error message
            placeholder.error(answer)

    # Add the final assistant's response to the session state
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Call the scroll-to-bottom function
    scroll_to_bottom()