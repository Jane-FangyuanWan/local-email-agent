import ollama
import imaplib
import email
from email.header import decode_header
import os

# ================= Configuration Area (Please modify this) =================
EMAIL_USER = os.getenv("EMAIL_USER", "")      # Loaded from .env
EMAIL_PASS = os.getenv("EMAIL_PASS", "")      # Loaded from .env
# ======================================================

def get_latest_email():
    print("Connecting to Gmail...")
    
    # 1. Connect to Gmail server (IMAP protocol)
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(EMAIL_USER, EMAIL_PASS) # Login
    except Exception as e:
        print(f"❌ Login failed! Please check your username and app password.\nError info: {e}")
        return None

    # 2. Open inbox
    mail.select("inbox")

    # 3. Search emails (Here we only look for the latest one)
    # 'ALL' means search all, then we pick the last one
    status, messages = mail.search(None, "ALL")
    email_ids = messages[0].split()
    
    if not email_ids:
        print("📭 Inbox is empty!")
        return None

    # 4. Fetch the content of the latest email
    latest_id = email_ids[-1]
    print(f"📩 Reading the latest email (ID: {latest_id.decode()})...")
    
    # fetch raw email data
    status, msg_data = mail.fetch(latest_id, "(RFC822)")
    
    raw_email = msg_data[0][1]
    msg = email.message_from_bytes(raw_email)

    # 5. Parse subject (Handle potential encoding issues)
    subject, encoding = decode_header(msg["Subject"])[0]
    if isinstance(subject, bytes):
        subject = subject.decode(encoding if encoding else "utf-8")
    
    sender = msg.get("From")
    
    # 6. Parse body (This step is tricky because the email might be plain text or HTML)
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain": # We only read the plain text part
                try:
                    body = part.get_payload(decode=True).decode()
                    break
                except:
                    pass
    else:
        body = msg.get_payload(decode=True).decode()

    mail.logout()
    
    # Return processed text
    return f"Subject: {subject}\nFrom: {sender}\nBody: {body[:800]}" # Only take the first 800 chars to avoid overwhelming the model

# --- Main Program Starts ---

print("🚀 Starting Agent...")

# 1. Fetch real email
real_email = get_latest_email()

if real_email:
    print("-" * 30)
    print(f"[Successfully read]\n{real_email[:200]}...") # Print first 200 characters to show
    print("-" * 30)

    # 2. Construct Prompt
    prompt = f"""
    You are a smart email assistant. Analyze this email:
    
    {real_email}
    
    Tasks:
    1. Classify it: [Urgent, School, Promo, Finance, Spam]
    2. Draft a short, professional reply (if needed).
    
    Output Format:
    Category: ...
    Reply: ...
    """

    # 3. Let Ollama ponder
    print("🧠 AI is analyzing...")
    response = ollama.chat(model='llama3.2', messages=[
      {'role': 'user', 'content': prompt},
    ])

    print("\n" + "="*10 + " AI Analysis Result " + "="*10)
    print(response['message']['content'])