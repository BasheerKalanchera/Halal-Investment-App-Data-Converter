import streamlit as st
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import requests

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import streamlit.components.v1 as components
import time

# --- Load .env for local development ---
load_dotenv()

# --- HIDE STREAMLIT STYLE ---
hide_streamlit_style = """
<style>
    [data-testid="stDecoration"] {
        display: none;
    }
    [data-testid="appCreatorAvatar"] {
        display: none !important;
    }
    a[href="https://streamlit.io/cloud"] {
        display: none !important;
    }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# --- END HIDE STYLE ---

def scroll_to_bottom():
    """A small JavaScript component that scrolls the chat container to the bottom."""
    js = f"""
    <script>
        function scroll(dummy_var_to_force_re_execution) {{
            var chat_container = parent.document.querySelector('div[data-testid="stChatInput"]');
            if (chat_container) {{
                chat_container.scrollIntoView({{behavior: "smooth"}});
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
GITHUB_READ_TOKEN = os.getenv("GITHUB_READ_TOKEN")

# --- Helper function to get gspread client ---
@st.cache_resource
def get_gspread_client():
    """Authenticates and returns a gspread client."""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(GCP_SA, scope)
    client = gspread.authorize(creds)
    return client

# --- Logging and Chat History Functions ---
def log_unanswered_to_google_sheets(user_query, user_id="Anonymous"):
    try:
        client = get_gspread_client()
        sheet = client.open("HalalGold_UnansweredLogs").sheet1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([timestamp, user_id, user_query])
    except Exception as e:
        st.warning(f"⚠️ Logging to Google Sheets failed: {e}")

def load_chat_history(user_id):
    """Loads chat history for a given user_id from Google Sheets."""
    if not user_id:
        return []
    try:
        client = get_gspread_client()
        sheet = client.open("HalalGold_UnansweredLogs").worksheet("ChatHistory")
        all_records = sheet.get_all_records()
        user_history = []
        for record in all_records:
            if record["UserID"] == user_id:
                user_history.append({"role": record["Role"], "content": record["Content"]})
        return user_history
    except gspread.exceptions.WorksheetNotFound:
        st.warning("ChatHistory sheet not found. A new one will be created if you start chatting.")
        return []
    except Exception as e:
        st.error(f"Failed to load chat history: {e}")
        return []

def save_message_to_history(user_id, role, content):
    """Saves a single message to the Google Sheets history."""
    if not user_id:
        return
    try:
        client = get_gspread_client()
        sheet = client.open("HalalGold_UnansweredLogs").worksheet("ChatHistory")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([timestamp, user_id, role, content])
    except Exception as e:
        st.warning(f"⚠️ Could not save message to history: {e}")

# --- Embedding model ---
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

# --- Retriever setup ---
@st.cache_resource(show_spinner=False)
def create_retriever_from_github():
    with st.spinner("Loading up your Aisar - Halal Gold investment assistant... Please wait."):
        _embeddings = load_embedding_model()
        try:
            headers = {"Authorization": f"token {GITHUB_READ_TOKEN}"} if GITHUB_READ_TOKEN else {}
            faiss_url = f"{GITHUB_RAW_URL}/{FAISS_DIR}/index.faiss"
            pkl_url = f"{GITHUB_RAW_URL}/{FAISS_DIR}/index.pkl"

            faiss_data = requests.get(faiss_url, headers=headers).content
            pkl_data = requests.get(pkl_url, headers=headers).content

            os.makedirs(FAISS_DIR, exist_ok=True)
            with open(os.path.join(FAISS_DIR, "index.faiss"), "wb") as f:
                f.write(faiss_data)
            with open(os.path.join(FAISS_DIR, "index.pkl"), "wb") as f:
                f.write(pkl_data)
            
            vectorstore = FAISS.load_local(FAISS_DIR, _embeddings, allow_dangerous_deserialization=True)
            return vectorstore.as_retriever(search_kwargs={"k": 10})

        except Exception as e:
            st.error(f"❌ Failed to download or load knowledge base from GitHub: {e}")
            st.stop()

# --- Load LLM and retriever ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
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
    "**Disclaimer:** This tool is not intended as investment or Shariah advice. Users should not rely on its responses for financial decisions and are advised to consult with a registered financial advisor or qualified Shariah scholar."
)

# --- User ID and Chat History Loading ---
with st.sidebar:
    st.header("User Details")
    # Use a key for the text_input to ensure its value persists in session_state
    st.text_input(
        "Enter your name or email to save/load history:",
        key="user_id"
    )
    # Button to explicitly load history
    if st.button("Load Chat History"):
        if st.session_state.user_id:
            st.session_state.messages = load_chat_history(st.session_state.user_id)
            st.rerun() # Rerun the script to display the loaded messages
        else:
            st.warning("Please enter a User ID to load history.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display a welcome message if the chat is empty
if not st.session_state.messages:
     st.session_state.messages.append(
            {"role": "assistant", "content": "Hello! Enter your User ID and click 'Load Chat History' to see your past conversations. Otherwise, just ask me a question to begin."}
     )

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if user_query := st.chat_input("Enter your question about Halal ways to invest in Gold here:"):
    if not st.session_state.user_id:
        st.warning("Please enter a name or email in the sidebar to start a new chat and save your history.")
        st.stop()

    # Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": user_query})
    save_message_to_history(st.session_state.user_id, "user", user_query)
    with st.chat_message("user"):
        st.markdown(user_query)

    # Process and display assistant's response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            placeholder.markdown("Searching and generating answer...")
            result = qa_chain.invoke({"query": user_query})
            answer = result["result"]
            placeholder.markdown(answer)

            fallback_message = "I don't have enough information"
            if fallback_message in answer:
                log_unanswered_to_google_sheets(
                    user_query.strip(), st.session_state.user_id
                )
        except Exception as e:
            answer = f"An error occurred: {e}"
            placeholder.error(answer)

    # Add assistant response to state and save it
    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_message_to_history(st.session_state.user_id, "assistant", answer)
    
    scroll_to_bottom()