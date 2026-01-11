import os
import base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup

# --- Configuration ---
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly', 
    'https://www.googleapis.com/auth/userinfo.email',
    'openid'
]
DB_CONNECTION = "postgresql+psycopg2://postgres:secret@localhost:5432/postgres"
COLLECTION_NAME = "email_vectors"
TOKEN_FILE = 'token.json'

def authenticate_gmail_and_get_email():
    """Handles Google Login with error recovery."""
    creds = None
    
    # 1. Load existing token
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception:
            os.remove(TOKEN_FILE)
            creds = None

    # 2. Refresh or Login if needed
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                if os.path.exists(TOKEN_FILE): os.remove(TOKEN_FILE)
                creds = None
        
        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
            
    service = build('gmail', 'v1', credentials=creds)
    profile = service.users().getProfile(userId='me').execute()
    return service, profile['emailAddress']

def parse_parts(parts):
    """Recursively extracts text from email parts."""
    text_content = ""
    for part in parts:
        mime_type = part.get('mimeType', '')
        if mime_type == 'text/plain' and 'data' in part['body']:
            data = part['body']['data']
            text_content += base64.urlsafe_b64decode(data).decode() + "\n"
        elif mime_type == 'application/pdf':
            text_content += f"\n[Attachment: {part.get('filename')} found]\n"
        if 'parts' in part:
            text_content += parse_parts(part['parts'])
    return text_content

def ingest_emails_for_user(limit=50):
    service, user_email = authenticate_gmail_and_get_email()
    print(f"--- Authenticated as: {user_email} ---")
    
    # --- SMART FILTER: Everything EXCEPT Spam/Promotions ---
    # This ensures we get 'Updates' (like Shivang's email) but not 'Promotions' junk
    results = service.users().messages().list(
        userId='me', 
        maxResults=limit, 
        q='-category:promotions -category:social -category:forums -in:spam -in:trash' 
    ).execute()
    
    messages = results.get('messages', [])
    docs = []
    print(f"Processing {len(messages)} emails for {user_email}...")
    
    for msg in messages:
        try:
            txt = service.users().messages().get(userId='me', id=msg['id']).execute()
            payload = txt['payload']
            headers = payload['headers']
            
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "No Subject")
            sender = next((h['value'] for h in headers if h['name'] == 'From'), "Unknown")
            date = next((h['value'] for h in headers if h['name'] == 'Date'), "Unknown")
            msg_id = txt['id']
            link = f"https://mail.google.com/mail/u/0/#all/{msg_id}"
            
            body_text = ""
            if 'parts' in payload:
                body_text = parse_parts(payload['parts'])
            elif 'body' in payload and 'data' in payload['body']:
                data = payload['body']['data']
                body_text = base64.urlsafe_b64decode(data).decode()
            
            soup = BeautifulSoup(body_text, "html.parser")
            clean_text = soup.get_text(separator=' ')
            
            metadata = {
                "user_id": user_email, 
                "subject": subject,
                "sender": sender,
                "date": date,
                "link": link
            }
            
            full_content = f"Link: {link}\nFrom: {sender}\nDate: {date}\nSubject: {subject}\n\n{clean_text}"
            docs.append(Document(page_content=full_content, metadata=metadata))
            
        except Exception as e:
            print(f"Error processing email: {e}")

    if docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=DB_CONNECTION,
            use_jsonb=True,
        )
        vector_store.add_documents(splits)
        return user_email, len(docs)
    
    return user_email, 0