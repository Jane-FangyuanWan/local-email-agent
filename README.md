# Local-First AI Email Agent
**CS5100 Final Project**

## Project Overview
The Local-First AI Email Agent is a privacy-centric automation pipeline. Unlike traditional email assistants that leak sensitive personal data to cloud APIs (like OpenAI GPT-4), this system operates entirely locally on consumer hardware. 

It fetches emails via IMAP, securely parses MIME payloads, evaluates implicit intent using Small Language Models (SLMs) bound by strict Pydantic JSON parameters, and allows users to authorize AI-drafted responses via a Streamlit "Human-in-the-loop" UI over SMTP.

To demonstrate domain adaptation superiority over zero-shot prompting, the project also features a custom Qwen-0.5B LoRA adapter fine-tuned on a bespoke synthetic dataset `dataset.json`.

---

## Architecture & Features

The system follows a containerized microservices architecture:
1. **Frontend (Streamlit):** An interactive dashbboard providing a "Human-in-the-loop" approval process, bypassing the risk of hallucinated autonomous actions.
2. **Backend (FastAPI):** Orchestrates IMAP/SMTP connections and Pydantic-constrained logic loops.
3. **Database (PostgreSQL):** A persistent state ledger ensuring emails aren't endlessly scanned and hallucinated duplicates aren't drafted.
4. **AI Inference Engine (Ollama):** Local Llama 3.2 execution bound mathematically to output `EmailAnalysis` schema structures.
5. **PEFT Logic:** Separate PyTorch/Transformers pipelines (`finetune.py`, `evaluate.py`) showcasing state-of-the-art Low Rank Adaptation fine-tuning.

---

## How to Run the App (Docker Pipeline)

### prerequisites
- A running instance of **Docker Desktop**.
- A local installation of [Ollama](https://ollama.com/) with Llama 3.2 pulled (`ollama run llama3.2`).

### 1. Environment Variables
Create a `.env` file in the root directory and ensure `EMAIL_USER` and `EMAIL_PASS` (Gmail App Password) are populated.
```ini
EMAIL_USER=your_email@gmail.com
EMAIL_PASS=your_app_password
DATABASE_URL=postgresql://user:password@db:5432/emails_db
```

### 2. Boot the Subsystems
Launch the architecture using Docker Compose. The script rebuilds the FastAPI and Streamlit images and hooks them into the PostgreSQL network.
```bash
docker-compose up --build
```

### 3. Usage
- Wait for the API container to establish `Database Connected!`.
- Navigate to the frontend at `http://localhost:8501`.
- Click the **Scan New Emails** button.
- Select scanned emails from the datatable to open the Draft UI. Review, Edit, and hit **Approve & Send**.

---

## Running the Benchmark Evaluator
As detailed in the `Final_Paper.pdf` submission, the project contains a benchmarking script to demonstrate the execution differences between Zero-Shot SLMs and LoRA-adapted ones.

To run the script natively (requires PyTorch, PEFT, Transformers, and Trl):
```bash
python3 evaluate.py
```
*Note: Depending on whether you are executing on Apple MPS or Nvidia Cuda, adjust the device routing explicitly if latency is unusually high.*

---

## Repository Map

- `/my_custom_email_adapter/` -> Custom trained state-dict (LoRA matrices)
- `dataset.json` -> Synthetic intent classification training data.
- `backend.py` -> FastAPI core router and IMAP traversal.
- `frontend.py` -> Streamlit interface layer.
- `finetune.py` -> PyTorch training logic for domain adaptation.
- `evaluate.py` -> Comparative grading mechanism benchmarking Latency and Structural Adherence.
- `Final_Paper.pdf` -> 10-page academic methodology exploration.