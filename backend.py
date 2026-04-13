import os
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
import imaplib
import email
from email.header import decode_header
import ollama
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, Field
import json
import smtplib
from email.mime.text import MIMEText
app = FastAPI()

# ================= Configuration Area =================
EMAIL_USER = os.getenv("EMAIL_USER", "")  # Uses value from .env file
EMAIL_PASS = os.getenv("EMAIL_PASS", "")  # Uses value from .env file
# Read database connection from environment variables (configured in docker-compose.yml)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/emails_db")
# ==========================================

# --- 1. Database Setup ---
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define schema
class EmailModel(Base):
    __tablename__ = "emails"
    
    id = Column(Integer, primary_key=True, index=True)
    subject = Column(String)
    sender = Column(String)
    body = Column(Text)
    category = Column(String) # AI-analyzed category
    action = Column(String)   # AI-suggested action (Reply/Ignore)
    draft = Column(Text)      # AI-generated draft reply

# Automatically create tables (if they do not exist)
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 2. Helper Functions ---

def get_gmail_connection():
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select("inbox")
        return mail
    except Exception as e:
        print(f"Login failed: {e}")
        return None

def parse_email_content(msg_data):
    raw_email = msg_data[0][1]
    msg = email.message_from_bytes(raw_email)
    subject, encoding = decode_header(msg["Subject"])[0]
    if isinstance(subject, bytes):
        subject = subject.decode(encoding if encoding else "utf-8")
    sender = msg.get("From")
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try: body = part.get_payload(decode=True).decode(); break
                except: pass
    else:
        body = msg.get_payload(decode=True).decode()
    return subject, sender, body[:500]

class EmailAnalysis(BaseModel):
    category: str = Field(description="[Finance/Academic/Personal/Spam/General]")
    action: str = Field(description="[Reply/Ignore]")
    draft: str = Field(description="[If the action is Reply, provide a short, professional draft response here. Otherwise, return an empty string]")

def ask_ollama(subject, sender, body):
    # Enforce structured output using Pydantic JSON Schema
    prompt = f"""
    Analyze this email.
    Sender: {sender}
    Subject: {subject}
    Body: {body}
    """
    try:
        response = ollama.chat(
            model='llama3.2', 
            messages=[{'role': 'user', 'content': prompt}], 
            format=EmailAnalysis.model_json_schema()
        )
        content = response['message']['content']
        
        # Parse the JSON string reliably
        data = json.loads(content)
        category = data.get("category", "General")
        action = data.get("action", "Ignore")
        draft = data.get("draft", "")
        return category, action, draft
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return "Error", "Ignore", ""

# --- 3. Endpoints ---

@app.get("/")
def home():
    return {"message": "Database Connected!"}

# Feature A: Scan and save to database
@app.get("/scan")
def scan_inbox(limit: int = 3, db: Session = Depends(get_db)):
    mail = get_gmail_connection()
    if not mail: return {"error": "Gmail connection failed"}

    status, messages = mail.search(None, "ALL")
    email_ids = messages[0].split()
    recent_ids = list(reversed(email_ids[-limit:]))
    
    saved_count = 0
    for e_id in recent_ids:
        status, msg_data = mail.fetch(e_id, "(RFC822)")
        subject, sender, body = parse_email_content(msg_data)
        
        # Deduplication: do not analyze if already in the database
        existing_email = db.query(EmailModel).filter(EmailModel.subject == subject).first()
        if not existing_email:
            # AI Inference
            cat, act, draft = ask_ollama(subject, sender, body)
            
            # Save to database!!
            new_email = EmailModel(subject=subject, sender=sender, body=body, category=cat, action=act, draft=draft)
            db.add(new_email)
            db.commit()
            saved_count += 1
            print(f"✅ Saved: {subject}")
        else:
            print(f"⏩ Skipped (Already exists): {subject}")

    mail.logout()
    return {"status": "success", "new_emails_saved": saved_count}

# Feature B: View all history records (The purpose of having a database!)
@app.get("/emails")
def get_all_emails(db: Session = Depends(get_db)):
    # SELECT * FROM emails;
    return db.query(EmailModel).all()

# [New] Feature: Update a specific email's draft
class DraftUpdate(BaseModel):
    draft: str

@app.put("/emails/{email_id}/draft")
def update_draft(email_id: int, payload: DraftUpdate, db: Session = Depends(get_db)):
    email_record = db.query(EmailModel).filter(EmailModel.id == email_id).first()
    if not email_record:
        raise HTTPException(status_code=404, detail="Email not found")
    
    email_record.draft = payload.draft
    db.commit()
    return {"status": "success", "message": "Draft updated successfully"}

# ... Previous code ...

# [New] Feature C: Delete a specific email
from fastapi import HTTPException # Remember to import this at the top of the file

@app.delete("/emails/{email_id}")
def delete_email(email_id: int, db: Session = Depends(get_db)):
    # 1. Find email
    email_to_delete = db.query(EmailModel).filter(EmailModel.id == email_id).first()
    
    # 2. Raise error if not found
    if not email_to_delete:
        raise HTTPException(status_code=404, detail="Email not found")
    
    # 3. Delete and commit
    db.delete(email_to_delete)
    db.commit()
    
    return {"status": "success", "message": f"Deleted email ID {email_id}"}


# [New] Feature D: Send a reply (Outgoing Email Module)
@app.post("/emails/{email_id}/send")
def send_reply(email_id: int, db: Session = Depends(get_db)):
    # 1. Find email
    email_record = db.query(EmailModel).filter(EmailModel.id == email_id).first()
    
    # 2. Raise error if not found
    if not email_record:
        raise HTTPException(status_code=404, detail="Email not found")
    if not email_record.draft:
        raise HTTPException(status_code=400, detail="No draft to send")
    
    # 3. Send via SMTP
    try:
        from_email = EMAIL_USER
        password = EMAIL_PASS
        to_email = email_record.sender
        
        msg = MIMEText(email_record.draft)
        msg['Subject'] = f"Re: {email_record.subject}"
        msg['From'] = from_email
        msg['To'] = to_email
        
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        
        return {"status": "success", "message": f"Reply sent to {to_email}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))