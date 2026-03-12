import streamlit as st
import requests
import pandas as pd

# Set page title
st.set_page_config(page_title="Email AI Agent", layout="wide")
st.title("🤖 Local Email AI Agent")

# Backend API URL (Docker internal network domain)
# Note: frontend runs in a container, use 'backend' to access backend, not localhost
API_URL = "http://backend:8000"

# --- Sidebar: Control Panel ---
with st.sidebar:
    st.header("Control Panel")
    
    # 1. Scan button (Keep the original functionality)
    if st.button("🔄 Scan New Emails"):
        with st.spinner("Scanning Inbox & Analyzing with AI..."):
            try:
                # Send request to backend
                response = requests.get(f"{API_URL}/scan?limit=5")
                data = response.json()
                
                # Show success message
                st.success(f"Scan Complete! Saved {data.get('new_emails_saved', 0)} new emails.")
                
                # Crucial: automatically refresh page after scan so new emails show up in the table immediately
                import time
                time.sleep(1) 
                st.rerun() 
            except Exception as e:
                st.error(f"Connection Failed: {e}")

    st.divider() # Draw a divider to make it look better

    # 2. Delete functional area [This is newly added!]
    st.subheader("🗑 Manage Emails")
    
    # Let user input ID
    delete_id = st.number_input("Enter Email ID to delete:", min_value=1, step=1)
    
    # Delete button
    if st.button("Delete Email"):
        try:
            # Send DELETE request to backend
            res = requests.delete(f"{API_URL}/emails/{delete_id}")
            
            if res.status_code == 200:
                st.success(f"Email {delete_id} deleted!")
                # Pause for 1 second to let user see the green success prompt, then refresh
                import time
                time.sleep(1) 
                st.rerun()    
            else:
                st.error("Error: Could not find that ID.")
        except Exception as e:
            st.error(f"Connection failed: {e}")

# --- Main Page: Display Data ---
st.subheader("📬 Inbox Intelligence")

# Automatically load data
try:
    response = requests.get(f"{API_URL}/emails")
    emails = response.json()
    
    if emails:
        # Convert to DataFrame for display
        df = pd.DataFrame(emails)
        
        # Simple beautification for table display
        # We only display crucial columns
        display_df = df[["id", "subject", "sender", "category", "action", "body"]]
        
        st.dataframe(
            display_df,
            column_config={
                "subject": st.column_config.TextColumn("Subject", width="medium"),
                "body": st.column_config.TextColumn("Content Preview", width="large"),
                "category": st.column_config.TextColumn("Category", width="small"),
                "action": st.column_config.TextColumn("Action", width="small"),
            },
            hide_index=True,
        )
    else:
        st.info("No emails found in database. Click 'Scan New Emails' to start.")
        
except Exception as e:
    st.error(f"Could not fetch data from backend. Is it running? \nError: {e}")