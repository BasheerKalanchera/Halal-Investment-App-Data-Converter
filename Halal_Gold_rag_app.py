import streamlit as st
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import requests
from operator import itemgetter

# --- Core LangChain and LangChain Community Imports ---
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

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
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    try:
        gcp_sa = st.secrets["gcp_service_account"]
    except Exception:
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

# --- Custom Chat History Class for Google Sheets ---
class GspreadChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.client = get_gspread_client()
        self.sheet = self.client.open("HalalGold_UnansweredLogs").worksheet("ChatHistory")

    @property
    def messages(self):
        """
        Efficiently retrieves chat messages for a specific session_id
        by finding matching rows and then fetching their content in a batch.
        """
        messages = []
        try:
            # Find all cells in the second column (B) that match the session_id
            user_cells = self.sheet.findall(self.session_id, in_column=2)
        except gspread.exceptions.CellNotFound:
            # If no cells are found for the user, return an empty history
            return []

        # Create a list of A1 notation ranges to fetch all rows at once
        ranges_to_get = [f'A{cell.row}:D{cell.row}' for cell in user_cells]

              
        if not ranges_to_get:
            return []
            
        # Get all row data in a single batch API call, which is highly efficient
        all_row_data = self.sheet.batch_get(ranges_to_get)

        # Process the batch data
        for row_group in all_row_data:
            for row in row_group:
                # Assuming columns are: Timestamp(A), UserID(B), Role(C), Content(D)
                if len(row) >= 4:
                    role = row[2]
                    content = row[3]
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
        return messages

    def add_messages(self, messages):
        rows_to_add = []
        for message in messages:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            rows_to_add.append([timestamp, self.session_id, role, message.content])
        if rows_to_add:
            self.sheet.append_rows(rows_to_add)

    def clear(self):
        pass

# --- Model and Retriever Loading ---
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

@st.cache_resource(show_spinner="Loading knowledge base...")
def create_retriever_from_github():
    FAISS_DIR = "faiss_index"
    GITHUB_RAW_URL = "https://raw.githubusercontent.com/BasheerKalanchera/Halal-Investment-App-Data-Converter/main"
    GITHUB_READ_TOKEN = os.getenv("GITHUB_READ_TOKEN")
    _embeddings = load_embedding_model()
    try:
        headers = {"Authorization": f"token {GITHUB_READ_TOKEN}"} if GITHUB_READ_TOKEN else {}
        faiss_url = f"{GITHUB_RAW_URL}/{FAISS_DIR}/index.faiss"
        pkl_url = f"{GITHUB_RAW_URL}/{FAISS_DIR}/index.pkl"
        faiss_data = requests.get(faiss_url, headers=headers).content
        pkl_data = requests.get(pkl_url, headers=headers).content
        os.makedirs(FAISS_DIR, exist_ok=True)
        with open(os.path.join(FAISS_DIR, "index.faiss"), "wb") as f: f.write(faiss_data)
        with open(os.path.join(FAISS_DIR, "index.pkl"), "wb") as f: f.write(pkl_data)
        vectorstore = FAISS.load_local(FAISS_DIR, _embeddings, allow_dangerous_deserialization=True)
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"❌ Failed to load knowledge base: {e}")
        st.stop()

# --- LCEL Chain Definitions ---
MODEL_NAME = None
try:
    MODEL_NAME = st.secrets["MODEL_NAME"]
except (KeyError, FileNotFoundError):
    MODEL_NAME = os.getenv("MODEL_NAME")

if not MODEL_NAME:
    st.warning("MODEL_NAME not found in secrets or .env. Using default: gemini-1.5-pro")
    MODEL_NAME = "gemini-1.5-pro" # Using a more capable model by default


llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0,max_output_tokens=2048)
retriever = create_retriever_from_github()

_condense_question_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and a follow-up question, rephrase the follow-up question to be a standalone question."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])
condense_question_chain = _condense_question_prompt | llm | StrOutputParser()

_answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Aisar, a helpful and stateful AI assistant for Halal Gold Investments. Your primary goal is to answer user questions based on the provided CONTEXT.

However, you have a special capability: you MUST also use the CHAT HISTORY to answer questions about the conversation itself.

Here are your strict rules:
1.  For questions about Halal Gold, use the CONTEXT. If the answer isn't in the CONTEXT, say so.
2.  If the user asks for a summary, a list of their past questions, or anything about the conversation, you MUST analyze the CHAT HISTORY to provide the answer.
3.  **DO NOT**, under any circumstances, say you are "stateless" or "cannot access chat history." Your function in this application is to be stateful by using the history provided below.
4. Keep ALL answers concise enough to fit into one Google Sheets cell:
   - Maximum response length: ~8,000 characters total.
   - When presenting tables: maximum 8 rows.
   - Each cell of a table must be under ~2,000 characters.
   - Do not repeat the same table multiple times.
   - Be clear and to the point.

CONTEXT:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])
answer_chain = _answer_prompt | llm | StrOutputParser()

def _format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# --- UPDATED CONVERSATIONAL CHAIN LOGIC ---
def route_question(input_dict):
    """
    If there is chat history, condense the question.
    Otherwise, use the original question for retrieval.
    """
    if input_dict.get("chat_history"):
        return condense_question_chain
    else:
        return itemgetter("question")

context_retrieval_chain = RunnablePassthrough.assign(
    standalone_question=route_question
) | {
    "context": itemgetter("standalone_question") | retriever | _format_docs,
    "question": itemgetter("question"), # Explicitly pass the original question through
    "chat_history": itemgetter("chat_history")
}

conversational_rag_chain = context_retrieval_chain | answer_chain

# --- Main App UI ---
st.set_page_config(page_title="Halal Gold Investment Assistant", layout="wide")
#st.title("💰 Aisar - Halal Gold Investment Assistant")
# --- HEADER SECTION ---
col1, col2 = st.columns([1, 12])  # Creates two columns for alignment

with col1:
    # Display your logo in the first column
   st.image("assets/logo.png", width=150)

with col2:
    # Display your title in the second column (without the emoji)
   st.title("- Halal Gold Investment Assistant")

st.warning(
    "**Disclaimer:** This tool is not intended as investment or Shariah advice. Users should not rely on its responses for financial decisions and are advised to consult with a registered financial advisor or qualified Shariah scholar."
)

# --- Callback function to load history into the UI session state ---
def load_history_for_ui():
    if st.session_state.get("user_id"):
        history = GspreadChatMessageHistory(st.session_state.user_id)
        st.session_state.messages = history.messages

# --- User ID and History Display ---
with st.sidebar:
    st.header("User Details")
    st.text_input(
        "Enter your name or email to save/load history:",
        key="user_id",
        on_change=load_history_for_ui
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.session_state.messages.append(AIMessage(content="Hello! Enter your User ID to see past chats, or ask a question to begin."))

for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# --- Chat Input and Chain Invocation ---
if user_query := st.chat_input("Enter your question about Halal ways to invest in Gold here:"):
    if not st.session_state.user_id:
        st.warning("Please enter a User ID in the sidebar to start a conversation.")
        st.stop()

    st.session_state.messages.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        # --- Check for list questions intent ---
        if "list of questions" in user_query.lower() or "what have i asked" in user_query.lower():
            # Filter the messages already in session state (in memory)
            user_questions = [
                msg.content for msg in st.session_state.messages 
                if isinstance(msg, HumanMessage)
            ]
            
            if user_questions:
                # Create the formatted list, excluding the current "list my questions" query
                response_list = "\n".join([f"* {q}" for q in user_questions if "list of questions" not in q.lower() and "what have i asked" not in q.lower()])
                full_response = "Here is the list of questions you've asked:\n\n" + response_list
            else:
                full_response = "I don't have a record of any questions from you yet."
            placeholder.markdown(full_response)
        else:
            # --- This is now the production code ---
            chain_with_history = RunnableWithMessageHistory(
                conversational_rag_chain,
                lambda session_id: GspreadChatMessageHistory(session_id),
                input_messages_key="question",
                history_messages_key="chat_history",
            )
            config = {"configurable": {"session_id": st.session_state.user_id}}
            
            try:
                for chunk in chain_with_history.stream({"question": user_query}, config=config):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                full_response = "Sorry, I ran into an error. Please try again."
                placeholder.markdown(full_response)

    st.session_state.messages.append(AIMessage(content=full_response))