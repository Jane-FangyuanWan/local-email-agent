# Local-First AI Email Agent

## Project Overview
This project is a privacy-focused email assistant that runs entirely locally using Docker. It leverages a local LLM (Llama 3.2 via Ollama) to classify emails and draft replies without sending data to the cloud.

## Architecture
The application follows a microservices architecture orchestrated by Docker Compose:
1.  **Frontend:** Streamlit (UI for user interaction)
2.  **Backend:** FastAPI (Business logic + AI inference)
3.  **Database:** PostgreSQL (Persistent storage for email history)
4.  **AI Engine:** Ollama (Running on host machine)

## How to Run
1.  Ensure Docker Desktop and Ollama are running.
2.  Update `backend/backend.py` with your App Password.
3.  Run the command:
    ```bash
    docker-compose up --build
    ```
4.  Access the UI at `http://localhost:8501`.

## Tech Stack
* Python 3.9
* Docker & Docker Compose
* FastAPI & SQLAlchemy
* Streamlit
* PostgreSQL