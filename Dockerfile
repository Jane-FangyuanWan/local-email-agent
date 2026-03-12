# 1. Base Image: Use Python 3.9 slim version
FROM python:3.9-slim

# 2. Set Working Directory: Create /app folder in the container
WORKDIR /app

# 3. Copy dependencies list
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy source code: Put backend.py into the container
COPY backend.py .

# 6. Expose port: The container will listen on port 8000
EXPOSE 8000

# 7. Start command: Run this command automatically when the container starts
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]