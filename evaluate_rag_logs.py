import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import time
import json

# --- Load .env for local development ---
load_dotenv()

# --- Configuration ---
EVAL_SHEET_NAME = "RAG_EVALUATION_LOGS"
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-pro")
GCP_EVAL_FOLDER_ID = os.getenv("GCP_EVAL_FOLDER_ID")

# --- Column Index Mapping (1-based for gspread) ---
UUID_COL = 1
TIMESTAMP_COL = 2
QUERY_COL = 3
DOC_IDS_COL = 4
CONTEXT_COL = 5
ANSWER_COL = 6
FAITHFULNESS_COL = 7
RELEVANCE_COL = 8

# --- GCP Service Account and gspread Client Setup ---
def get_gspread_client():
    """Establishes a connection to Google Sheets using credentials."""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    try:
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
            raise ValueError("GCP service account details not found in .env file.")
        creds = ServiceAccountCredentials.from_json_keyfile_dict(gcp_sa, scope)
        return gspread.authorize(creds)
    except Exception as e:
        print(f"Error connecting to Google Sheets: {e}")
        return None

# --- LLM-as-a-Judge Chains ---
class EvaluationResult(JsonOutputParser):
    """Parses the LLM's JSON output for evaluation results."""
    def parse(self, text: str):
        try:
            if text.strip().startswith("```json"):
                text = text.strip()[7:-3]
            data = json.loads(text)
            return {
                "score": data.get("score"),
                "justification": data.get("justification")
            }
        except json.JSONDecodeError:
            return {"score": 0, "justification": f"Failed to parse JSON: {text}"}

def get_faithfulness_evaluator():
    """Creates a LangChain chain to evaluate answer faithfulness."""
    prompt = ChatPromptTemplate.from_template(
        """SYSTEM: You are an expert AI evaluator. Your task is to assess the faithfulness of a generated answer based on a provided context.
        The answer is "faithful" if all claims made in the answer are directly supported by the information in the context.

        USER:
        CONTEXT:
        ---
        {context}
        ---

        ANSWER:
        ---
        {answer}
        ---

        INSTRUCTIONS:
        1. Compare the ANSWER to the CONTEXT.
        2. Determine if all information in the ANSWER is present in the CONTEXT.
        3. Provide a score from 1 to 5, where 1 is not faithful at all and 5 is perfectly faithful.
        4. Provide a brief justification for your score.
        5. Output your response as a JSON object with "score" and "justification" keys.

        Example JSON output:
        {{
            "score": 5,
            "justification": "The answer accurately summarizes the information provided in the context without adding any new claims."
        }}
        """
    )
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)
    return prompt | llm | EvaluationResult()

def get_relevance_evaluator():
    """Creates a LangChain chain to evaluate answer relevance."""
    prompt = ChatPromptTemplate.from_template(
        """SYSTEM: You are an expert AI evaluator. Your task is to assess the relevance of a generated answer to a user's question.
        The answer is "relevant" if it directly addresses and satisfactorily answers the user's question.

        USER:
        QUESTION:
        ---
        {question}
        ---

        ANSWER:
        ---
        {answer}
        ---

        INSTRUCTIONS:
        1. Compare the ANSWER to the QUESTION.
        2. Determine if the ANSWER directly addresses the user's intent.
        3. Ignore whether the answer is factually correct; focus only on its relevance to the question.
        4. Provide a score from 1 to 5, where 1 is not relevant at all and 5 is perfectly relevant.
        5. Provide a brief justification for your score.
        6. Output your response as a JSON object with "score" and "justification" keys.

        Example JSON output:
        {{
            "score": 5,
            "justification": "The answer directly addresses the user's question about Shariah compliance."
        }}
        """
    )
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)
    return prompt | llm | EvaluationResult()

# --- Main Evaluation Logic ---
def main():
    print("--- Starting RAG Evaluation Script ---")
    client = get_gspread_client()
    if not client:
        return

    try:
        print(f"Opening spreadsheet '{EVAL_SHEET_NAME}'...")
        if not GCP_EVAL_FOLDER_ID:
            raise ValueError("GCP_EVAL_FOLDER_ID not found in .env file.")
        
        spreadsheet = client.open(EVAL_SHEET_NAME, folder_id=GCP_EVAL_FOLDER_ID)
        sheet = spreadsheet.sheet1
        print("Spreadsheet opened successfully.")
    except Exception as e:
        print(f"Error: Could not open spreadsheet. Please ensure it exists in the specified folder and is shared with the service account. Details: {e}")
        return

    # --- UPDATED: Using a more robust method to find and process rows ---
    print("Fetching all records to find unprocessed rows...")
    all_records = sheet.get_all_records()

    evaluation_queue = []
    for i, record in enumerate(all_records):
        row_num = i + 2  # +2 because get_all_records skips the header, and sheets are 1-indexed

        # Use .get() to safely access keys that might not exist in empty rows
        f_score = record.get("Faithfulness Score", "")
        r_score = record.get("Answer Relevance Score", "")

        # Check if both score fields are empty (after stripping potential whitespace)
        if not str(f_score).strip() and not str(r_score).strip():
            evaluation_queue.append((row_num, record))

    if not evaluation_queue:
        print("No new rows to evaluate. All logs are up-to-date.")
        return

    print(f"Found {len(evaluation_queue)} new log(s) to evaluate.")
    
    faithfulness_evaluator = get_faithfulness_evaluator()
    relevance_evaluator = get_relevance_evaluator()

    for row_num, row_data in evaluation_queue:
        print(f"\n--- Evaluating Row {row_num} (UUID: {row_data.get('UUID')}) ---")
        question = row_data.get("User Query")
        context = row_data.get("Retrieved Context")
        answer = row_data.get("Generated Answer")

        if not all([question, context, answer]):
            print("Skipping row due to missing data.")
            continue
        
        # --- Faithfulness Evaluation ---
        print("Running Faithfulness evaluation...")
        faithfulness_result = faithfulness_evaluator.invoke({"context": context, "answer": answer})
        faithfulness_score = f"{faithfulness_result['score']}/5 - {faithfulness_result['justification']}"
        print(f"  -> Score: {faithfulness_score}")
        sheet.update_cell(row_num, FAITHFULNESS_COL, faithfulness_score)
        time.sleep(2) 

        # --- Relevance Evaluation ---
        print("Running Answer Relevance evaluation...")
        relevance_result = relevance_evaluator.invoke({"question": question, "answer": answer})
        relevance_score = f"{relevance_result['score']}/5 - {relevance_result['justification']}"
        print(f"  -> Score: {relevance_score}")
        sheet.update_cell(row_num, RELEVANCE_COL, relevance_score)
        time.sleep(2)

    print("\n--- Evaluation complete. All new logs have been scored. ---")

if __name__ == "__main__":
    main()





