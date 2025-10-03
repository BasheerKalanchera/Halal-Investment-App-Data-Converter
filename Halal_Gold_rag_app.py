import streamlit as st
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import requests
import uuid
from typing import TypedDict, Sequence, Any, List
import threading

# --- Core LangChain, LangGraph and Community Imports ---
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langgraph.graph import StateGraph, END

# --- LangChain Partner Package Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# --- Standard Library Imports ---
from dotenv import load_dotenv

# --- Load .env for local development ---
load_dotenv()

# --- HIDE STREAMLIT STYLE ---
st.markdown("""
<style>
    [data-testid="stDecoration"] {display: none;}
    [data-testid="appCreatorAvatar"] {display: none !important;}
    a[href="https://streamlit.io/cloud"] {display: none !important;}
</style>
""", unsafe_allow_html=True)


# --- GCP Service Account and gspread Client Setup ---
@st.cache_resource
def get_gspread_client():
    """Establishes a connection to Google Sheets using credentials."""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    try:
        # Try loading from Streamlit secrets first (for deployment)
        gcp_sa = st.secrets["gcp_service_account"]
    except Exception:
        # Fallback to .env file for local development
        gcp_sa = {
            "type": os.getenv("GCP_TYPE"), "project_id": os.getenv("GCP_PROJECT_ID"),
            "private_key_id": os.getenv("GCP_PRIVATE_KEY_ID"),
            "private_key": os.getenv("GCP_PRIVATE_KEY", "").replace("\\n", "\n"),
            "client_email": os.getenv("GCP_CLIENT_EMAIL"), "client_id": os.getenv("GCP_CLIENT_ID"),
            "auth_uri": os.getenv("GCP_AUTH_URI"), "token_uri": os.getenv("GCP_TOKEN_URI"),
            "auth_provider_x509_cert_url": os.getenv("GCP_AUTH_PROVIDER_X509_CERT_URL"),
            "client_x509_cert_url": os.getenv("GCP_CLIENT_X509_CERT_URL"),
        }
    if not all(gcp_sa.values()):
        st.error("GCP service account details not found. Please set them in Streamlit secrets or your .env file.")
        st.stop()
    creds = ServiceAccountCredentials.from_json_keyfile_dict(gcp_sa, scope)
    return gspread.authorize(creds)

# --- Chat History Management using Google Sheets ---
class GspreadChatMessageHistory(BaseChatMessageHistory):
    """Manages chat history for a session by reading from and writing to Google Sheets."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.client = get_gspread_client()
        self.sheet = self.client.open("HalalGold_UnansweredLogs").worksheet("ChatHistory")

    @property
    def messages(self) -> list[BaseMessage]:
        """Retrieve messages from Google Sheets for the current session."""
        messages = []
        try:
            user_cells = self.sheet.findall(self.session_id, in_column=2)
        except gspread.exceptions.CellNotFound:
            return [] # No history for this user yet
        ranges_to_get = [f'A{cell.row}:D{cell.row}' for cell in user_cells]
        if not ranges_to_get: return []
        all_row_data = self.sheet.batch_get(ranges_to_get)
        for row_group in all_row_data:
            for row in row_group:
                if len(row) >= 4:
                    role, content = row[2], row[3]
                    if role == "user": messages.append(HumanMessage(content=content))
                    elif role == "assistant": messages.append(AIMessage(content=content))
        return messages

    def add_messages(self, messages: list[BaseMessage]):
        """Append a list of new messages to the Google Sheet."""
        rows_to_add = []
        for message in messages:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            rows_to_add.append([timestamp, self.session_id, role, message.content])
        if rows_to_add: self.sheet.append_rows(rows_to_add)

    def clear(self):
        """Clearing history is not implemented to prevent data loss."""
        pass

# --- Model and Retriever Loading ---
@st.cache_resource(show_spinner=False)
def create_retriever_from_github():
    """Downloads FAISS index from GitHub and creates a retriever."""
    FAISS_DIR = "faiss_index"
    GITHUB_RAW_URL = "https://raw.githubusercontent.com/BasheerKalanchera/Halal-Investment-App-Data-Converter/main"
    GITHUB_READ_TOKEN = os.getenv("GITHUB_READ_TOKEN")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    try:
        headers = {"Authorization": f"token {GITHUB_READ_TOKEN}"} if GITHUB_READ_TOKEN else {}
        faiss_response = requests.get(f"{GITHUB_RAW_URL}/{FAISS_DIR}/index.faiss", headers=headers)
        pkl_response = requests.get(f"{GITHUB_RAW_URL}/{FAISS_DIR}/index.pkl", headers=headers)
        faiss_response.raise_for_status()
        pkl_response.raise_for_status()
        os.makedirs(FAISS_DIR, exist_ok=True)
        with open(os.path.join(FAISS_DIR, "index.faiss"), "wb") as f: f.write(faiss_response.content)
        with open(os.path.join(FAISS_DIR, "index.pkl"), "wb") as f: f.write(pkl_response.content)
        vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"❌ Failed to load knowledge base: {e}")
        st.stop()

# --- LangGraph State Definition ---
class RAGState(TypedDict):
    """Defines the state that flows through the LangGraph."""
    question: str
    chat_history: Sequence[BaseMessage]
    context: str
    answer: str
    log_id: str
    doc_ids: List[str]

# --- LOGICAL AGENTS (AS GRAPH NODES) ---

def retriever_node(state: RAGState):
    """Retrieval Agent: Fetches context and document IDs from the knowledge base."""
    print("---AGENT: RETRIEVER---")
    retriever = create_retriever_from_github()
    docs = retriever.invoke(state["question"])
    context = "\n\n".join(d.page_content for d in docs)
    doc_ids = [doc.metadata.get("source", "Unknown") for doc in docs]
    return {"context": context, "doc_ids": doc_ids}

def generator_node(state: RAGState):
    """Generator Agent: Creates an answer based on context and history."""
    print("---AGENT: GENERATOR---")
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-pro")
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, max_output_tokens=2048)
    # UPDATED: More nuanced prompt to encourage synthesis
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are Aisar, a helpful and stateful AI assistant for Halal Gold Investments.

Your most important instruction is to be a **Retrieval-Augmented Generation** bot. This means you must base your answers on the provided CONTEXT.

Here are your strict rules, in order of importance:
1.  **GROUNDING IS MANDATORY:** You MUST answer questions based *only* on the CONTEXT provided. Do not use any of your own general knowledge or information from outside the CONTEXT.
2.  **BE COMPREHENSIVE (SYNTHESIZE):** Synthesize your answer using all relevant information in the CONTEXT. If the CONTEXT does not fully answer the question, you must still provide the partial information that is available and then state what is missing. For example, if asked for a list of compliant products, and the context only mentions where to find the list and provides examples of non-compliant products, you must share both of these facts.
3.  **DO NOT HALLUCINATE:** Never invent information that is not in the CONTEXT. If the CONTEXT is completely silent on a topic, then you must state that you cannot answer based on the provided information.
4.  **USE CHAT HISTORY:** For questions about the conversation itself (e.g., summaries, past questions), you MUST analyze the CHAT HISTORY to provide the answer.
5.  **DO NOT BE EVASIVE:** Never say you are "stateless" or "cannot access chat history." Use the history provided.
6.  **BE CONCISE FOR GOOGLE SHEETS:** Keep all answers concise enough to fit into one Google Sheets cell (max ~8,000 chars, tables max 8 rows).

CONTEXT:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    
    return {"answer": chain.stream({
        "question": state["question"],
        "context": state["context"],
        "chat_history": state["chat_history"]
    }), "log_id": str(uuid.uuid4())}

def background_logger(state: RAGState):
    """Logging function to be run in a separate thread."""
    print("---BACKGROUND LOGGER: STARTED---")
    EVAL_SHEET_NAME = "RAG_EVALUATION_LOGS"
    
    try:
        eval_folder_id = os.getenv("GCP_EVAL_FOLDER_ID")
        if not eval_folder_id:
            print("GCP_EVAL_FOLDER_ID not found in environment variables. Please set it in your .env file.")
            return

        client = get_gspread_client()
        
        try:
            spreadsheet = client.open(EVAL_SHEET_NAME, folder_id=eval_folder_id)
        except gspread.exceptions.SpreadsheetNotFound:
            print(f"Evaluation sheet '{EVAL_SHEET_NAME}' not found. Please create it manually and share it with the service account.")
            return

        sheet = spreadsheet.sheet1
        retrieved_ids_str = "\n".join(state.get("doc_ids", []))

        log_entry = [
            state["log_id"],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            state["question"],
            retrieved_ids_str,
            state["context"],
            state["answer"],
            "",  
            ""   
        ]
        sheet.append_row(log_entry)
        print("---BACKGROUND LOGGER: COMPLETE---")
    except gspread.exceptions.GSpreadException as e:
        print(f"Gspread Error in background logger: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in background logger: {e}")

# --- Build the Graph ---
workflow = StateGraph(RAGState)
workflow.add_node("retriever", retriever_node)
workflow.add_node("generator", generator_node)
workflow.set_entry_point("retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", END)
app_graph = workflow.compile()


# --- Streamlit UI (The "Invoker" of the Graph) ---
#st.set_page_config(page_title="Halal Gold Investment Assistant", layout="wide")
#col1, col2 = st.columns([1, 12])
#with col1: st.image("assets/logo.png", width=150)
#with col2: st.title("Halal Gold Investment Assistant")

st.set_page_config(page_title="Halal Gold Investment Assistant", layout="wide")

# Inject custom CSS to center items in Streamlit's columns
st.markdown("""
    <style>
    /* This targets the container of st.columns */
    [data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Now, your original code will work as expected
col1, col2 = st.columns([1, 12])

with col1:
    st.image("assets/logo.png", width=150)

with col2:
    # We use st.markdown for the title to remove the default top margin of st.title
    st.markdown('<h1 style="margin: 0; padding: 0;">Halal Gold Investment Assistant</h1>', unsafe_allow_html=True)

st.markdown("---") # Add a separator

st.warning(
    "**Disclaimer:** This tool is not intended as investment or Shariah advice. Users should not rely on its responses for financial decisions and are advised to consult with a registered financial advisor or qualified Shariah scholar."
)

def load_history_for_ui():
    if st.session_state.get("user_id"):
        history = GspreadChatMessageHistory(st.session_state.user_id)
        st.session_state.messages = history.messages

with st.sidebar:
    st.header("User Details")
    st.text_input("Enter your name or email to save/load history:", key="user_id", on_change=load_history_for_ui)

if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hello! Enter your User ID to see past chats, or ask a question to begin.")]

for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role): st.markdown(message.content)

if user_query := st.chat_input("Enter your question about Halal ways to invest in Gold here:"):
    if not st.session_state.user_id:
        st.warning("Please enter a User ID in the sidebar to start a conversation.")
        st.stop()

    st.session_state.messages.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("▌") 

        # 1. ✅ Get history from session_state (fast) instead of Google Sheets (slow).
        initial_state = {
            "question": user_query, 
            "chat_history": st.session_state.get("messages", [])
        }

        try:
            full_response = ""
            final_log_state = initial_state.copy()
            stream_iterator = app_graph.stream(initial_state)
            
            # --- Stream processing and RAG logic (remains unchanged) ---
            retriever_event = next(stream_iterator)
            if 'retriever' in retriever_event:
                 final_log_state.update(retriever_event['retriever'])

            generator_event = next(stream_iterator)
            if 'generator' in generator_event:
                answer_stream = generator_event['generator']['answer']
                for chunk in answer_stream:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                
                final_log_state.update(generator_event['generator'])
                final_log_state['answer'] = full_response
            # --- End of stream processing ---

            # 2. ✅ Update the UI's message list in session_state first.
            st.session_state.messages.append(AIMessage(content=full_response))
            
            # 3. ✅ Now, create the history manager ONLY to write the new messages back to Sheets.
            history_manager = GspreadChatMessageHistory(st.session_state.user_id)
            history_manager.add_messages([
                HumanMessage(content=user_query), 
                AIMessage(content=full_response)
            ])

            # Start the background logger thread as before.
            if final_log_state:
                log_thread = threading.Thread(target=background_logger, args=(final_log_state,))
                log_thread.start()

        except Exception as e:
            st.error(f"An error occurred: {e}")
            response_placeholder.markdown("Sorry, I ran into an error. Please try again.")

