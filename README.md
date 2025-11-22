# AI-Powered-Workflow-Automation-Engine-RPA-ML-


Short description:
Automation engine to extract text from business documents (OCR), classify them with a simple ML model, and trigger RPA workflows to execute standard repetitive tasks (web form fill, validation). The backend exposes REST APIs (FastAPI) for document ingestion and job status.

Key features

Tesseract OCR wrapper for text extraction

ML-based document classifier (Scikit-learn pipeline)

RPA automation example using Selenium for browser tasks

FastAPI backend with endpoints to ingest document, get status, and fetch results

Containerized with Docker and docker-compose

SQLite for simple persistence (can be switched to MySQL/Postgres)

Architecture & Workflow (textual)

Client uploads a document via POST /ingest (file upload).

Server stores the file, extracts text via OCR (ocr.py).

Extracted text is sent to ML classifier (ml_model.py) to predict document type (invoice, receipt, form, etc).

Based on classification, a corresponding RPA workflow (rpa.py) is executed asynchronously (demo uses a synchronous call for clarity).

Results and extracted structured fields are saved to DB and returned to client.

Quickstart (local, development)

Prerequisites:

Python 3.10+

Tesseract OCR installed on host (for Debian/Ubuntu: sudo apt-get install tesseract-ocr)

Chrome + chromedriver for Selenium RPA OR use Playwright instead (chromedriver path configurable)

Docker (optional)

Clone repo and install dependencies:

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt


Train basic model (or use pre-saved models/doc_classifier.pkl):

python scripts/train_model.py --data sample_data/labels.csv --out models/doc_classifier.pkl


Run FastAPI:

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000


Ingest a sample file:

curl -F "file=@sample_data/invoices/inv1.png" http://localhost:8000/ingest


Check status:

GET /status/{job_id}

Docker (optional)

Build and run with docker-compose:

docker-compose up --build


This will create a container running the FastAPI app. Note: Tesseract must be available in the image (Dockerfile includes that).

Endpoints

POST /ingest — Upload file. Returns job_id.

form-data: file (binary)

GET /status/{job_id} — Get job status and results

GET /health — Health check

Example ingest response:

{
  "job_id": "job_20251122_001",
  "status": "processing"
}

File: requirements.txt
fastapi==0.95.2
uvicorn[standard]==0.22.0
pydantic==1.10.11
python-multipart==0.0.6
tesserocr==2.6.3   # optional; we will also support pytesseract
pillow==10.0.0
pytesseract==0.3.10
scikit-learn==1.3.0
joblib==1.2.0
sqlalchemy==2.1.0
alembic==1.11.1
selenium==4.12.0
requests==2.31.0
python-dotenv==1.0.0
pytest==7.4.0


If tesserocr gives trouble, pytesseract + PIL is used in the code.

Core code files

Copy the code below into the indicated paths.

app/schemas.py
from pydantic import BaseModel
from typing import Optional, Dict

class IngestResponse(BaseModel):
    job_id: str
    status: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    doc_type: Optional[str] = None
    extracted_text: Optional[str] = None
    structured_data: Optional[Dict] = None

app/db.py
from sqlalchemy import create_engine, Column, Integer, String, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DB_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True)
    status = Column(String, default="pending")
    doc_type = Column(String, nullable=True)
    extracted_text = Column(Text, nullable=True)
    structured_data = Column(JSON, nullable=True)

def init_db():
    Base.metadata.create_all(bind=engine)

app/utils.py
import os, uuid
from pathlib import Path

STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "storage"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

def save_uploaded_file(upload_file) -> str:
    ext = Path(upload_file.filename).suffix or ".bin"
    file_id = f"{uuid.uuid4().hex}{ext}"
    filepath = STORAGE_DIR / file_id
    with open(filepath, "wb") as f:
        content = upload_file.file.read()
        f.write(content)
    return str(filepath)

app/ocr.py
from PIL import Image
import pytesseract
import os
import tempfile

# Ensure tesseract binary accessible in PATH or set pytesseract.pytesseract.tesseract_cmd

def image_to_text(image_path: str) -> str:
    """Return OCR text for an image path (supports png/jpg/pdf first page)."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    elif ext == ".pdf":
        # For PDF, use PIL to open first page if possible (needs pdf2image in heavy use)
        from pdf2image import convert_from_path
        pages = convert_from_path(image_path, first_page=1, last_page=1)
        if pages:
            return pytesseract.image_to_string(pages[0])
    return ""

app/ml_model.py
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from typing import List

MODEL_PATH = os.getenv("MODEL_PATH", "models/doc_classifier.pkl")

def train_and_save(texts: List[str], labels: List[str], out_path=MODEL_PATH):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe.fit(texts, labels)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(pipe, out_path)
    return out_path

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def predict_text_class(model, text: str) -> str:
    if model is None:
        return "unknown"
    return model.predict([text])[0]

app/rpa.py
"""
Simple RPA example using Selenium to login and submit a form.
This is a demo; adapt selectors and steps to real workflows.
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import os

CHROME_DRIVER_PATH = os.getenv("CHROME_DRIVER_PATH", "chromedriver")  # set to absolute path if needed

def run_invoice_workflow_simulation(invoice_data: dict, headless=True) -> dict:
    # invoice_data: { "invoice_no": "...", "amount": "...", "vendor": "..." }
    options = Options()
    if headless:
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(executable_path=CHROME_DRIVER_PATH, options=options)
    result = {"status": "failed", "details": ""}
    try:
        # Replace with real client web URL and selectors
        driver.get("https://example.com/login")
        time.sleep(1)
        # login demo
        # driver.find_element(By.ID, "username").send_keys("user")
        # driver.find_element(By.ID, "password").send_keys("pass")
        # driver.find_element(By.ID, "login-btn").click()
        # time.sleep(2)
        # navigate to invoice form
        driver.get("https://example.com/invoice-form")
        time.sleep(1)
        # demo form fill (replace selectors)
        # driver.find_element(By.NAME, "invoice_no").send_keys(invoice_data.get("invoice_no", ""))
        # driver.find_element(By.NAME, "amount").send_keys(str(invoice_data.get("amount", "")))
        # driver.find_element(By.NAME, "vendor").send_keys(invoice_data.get("vendor", ""))
        # driver.find_element(By.ID, "submit").click()
        time.sleep(1)
        result["status"] = "success"
        result["details"] = "Simulated form submission (demo)"
    except Exception as e:
        result["details"] = str(e)
    finally:
        driver.quit()
    return result


Note: Selenium code above uses example.com placeholders — update URLs and selectors to suit your target RPA scenario. For headless Chrome in Docker, ensure Chrome and chromedriver installed in the image.

app/main.py
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.schemas import IngestResponse, JobStatus
from app.utils import save_uploaded_file
from app.ocr import image_to_text
from app.ml_model import load_model, predict_text_class
from app.db import SessionLocal, init_db, Job
import os
import threading
import time
from app.rpa import run_invoice_workflow_simulation

app = FastAPI(title="AI Workflow Automation Engine")

init_db()
MODEL = load_model()

def background_process_job(job_id: str, filepath: str):
    db = SessionLocal()
    job = db.query(Job).filter(Job.job_id == job_id).first()
    try:
        job.status = "processing"
        db.commit()
        # OCR
        text = image_to_text(filepath)
        job.extracted_text = text
        db.commit()
        # classify document
        doc_type = predict_text_class(MODEL, text) if MODEL else "unknown"
        job.doc_type = doc_type
        db.commit()
        # basic structured extraction demo (naive)
        structured = {}
        if "invoice" in doc_type.lower():
            # naive parse
            for line in text.splitlines():
                if "invoice" in line.lower() and "no" in line.lower():
                    structured["invoice_no"] = line.strip()
                if "total" in line.lower() or "amount" in line.lower():
                    structured["amount_line"] = line.strip()
        job.structured_data = structured
        db.commit()
        # Trigger corresponding RPA
        if doc_type.lower() == "invoice":
            # provide minimal fields
            invoice_data = {
                "invoice_no": structured.get("invoice_no", ""),
                "amount": structured.get("amount_line", "")
            }
            rpa_result = run_invoice_workflow_simulation(invoice_data, headless=True)
            job.status = "completed"
            job.structured_data = {**(job.structured_data or {}), "rpa_result": rpa_result}
            db.commit()
        else:
            job.status = "completed"
            db.commit()
    except Exception as e:
        job.status = "failed"
        job.structured_data = {"error": str(e)}
        db.commit()
    finally:
        db.close()

@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    # Save file
    filepath = save_uploaded_file(file)
    job_id = f"job_{uuid.uuid4().hex}"
    db = SessionLocal()
    new_job = Job(job_id=job_id, status="queued")
    db.add(new_job)
    db.commit()
    db.close()
    # Start background thread (simple approach)
    thread = threading.Thread(target=background_process_job, args=(job_id, filepath))
    thread.start()
    return IngestResponse(job_id=job_id, status="queued")

@app.get("/status/{job_id}", response_model=JobStatus)
def get_status(job_id: str):
    db = SessionLocal()
    job = db.query(Job).filter(Job.job_id == job_id).first()
    db.close()
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatus(
        job_id=job.job_id,
        status=job.status,
        doc_type=job.doc_type,
        extracted_text=job.extracted_text,
        structured_data=job.structured_data
    )

@app.get("/health")
def health():
    return {"status": "ok"}

scripts/train_model.py
import argparse
import pandas as pd
from app.ml_model import train_and_save

def load_labels(csv_path):
    # CSV with columns: filepath, label, text (optional)
    df = pd.read_csv(csv_path)
    texts = []
    labels = []
    for _, row in df.iterrows():
        if "text" in row and isinstance(row["text"], str) and row["text"].strip():
            texts.append(row["text"])
        else:
            # if text unavailable, skip or extract OCR externally
            texts.append("")
        labels.append(row["label"])
    return texts, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="models/doc_classifier.pkl")
    args = parser.parse_args()
    texts, labels = load_labels(args.data)
    train_and_save(texts, labels, out_path=args.out)
    print("Model trained and saved to", args.out)

scripts/process_document.py
import requests
def ingest_file(url: str, filepath: str):
    with open(filepath, "rb") as f:
        files = {"file": (filepath, f, "application/octet-stream")}
        r = requests.post(url + "/ingest", files=files)
        return r.json()

if __name__ == "__main__":
    url = "http://localhost:8000"
    print(ingest_file(url, "sample_data/invoices/inv1.png"))

Dockerfile
FROM python:3.10-slim

# Install tesseract and chrome dependencies for selenium headless
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libglib2.0-0 libnss3 libxss1 libasound2 libatk1.0-0 libgtk-3-0 \
    wget unzip gnupg --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Install Chrome & chromedriver (quick method; version pinning recommended)
RUN apt-get update && apt-get install -y wget unzip ca-certificates && \
    CHROME_VERSION=116.0.5845.96 && \
    wget -q -O /tmp/chrome.deb "https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb" || true

# If the above fails in some targets, you can provide your own chrome/chromedriver
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
ENV MODEL_PATH=/app/models/doc_classifier.pkl
ENV STORAGE_DIR=/app/storage
RUN mkdir -p /app/storage /app/models

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


Note: Chrome install above is simplified — in production pin versions carefully.

docker-compose.yml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./:/app
    environment:
      - DATABASE_URL=sqlite:///./app.db
      - MODEL_PATH=/app/models/doc_classifier.pkl
      - STORAGE_DIR=/app/storage
      - CHROME_DRIVER_PATH=/usr/local/bin/chromedriver

tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

Sample sample_data/labels.csv (format)
filepath,label,text
sample_data/invoices/inv1.png,invoice,"Invoice No: INV-001\nTotal: 123.45\nVendor: ABC Pvt Ltd"
sample_data/receipts/rcpt1.png,receipt,"Receipt for purchase\nTotal: 45.00"


You can create a small labeled CSV like above to train the basic classifier — OR populate the text column by running OCR offline and filling the CSV.

How to structure commits (suggested)

feat(api): add ingest + status endpoints

feat(ocr): add OCR wrapper

feat(ml): add training script and classifier

feat(rpa): add selenium demo

chore(docker): add Dockerfile and compose

Extending the project

Replace SQLite with MySQL/Postgres — update DATABASE_URL.

Improve ML pipeline: use transformer-based embeddings + fine-tuning (HuggingFace).

For production RPA, use robust orchestration (Celery/RabbitMQ / Kubernetes Jobs).

Add authentication (OAuth2) to APIs.

Add logging and observability (Prometheus + Grafana).

Add structured field extractors (rule-based + ML NER).

Sample usage and notes

Train or place a model at models/doc_classifier.pkl. If none present, classification returns "unknown".

Tesseract must be installed and available on PATH for pytesseract.

Selenium requires chromedriver — set CHROME_DRIVER_PATH env var to driver path.

Final notes

This starter implements the core flow (upload → OCR → classify → RPA simulate → persist). It is intentionally modular so you can replace individual pieces (e.g., swap Tesseract for commercial OCR, or replace LogisticRegression with an LLM embedding+classifier).

If you'd like, I can:

Create a complete GitHub repo archive (zipped) ready to push,

Flesh out production-grade Dockerfile with pinned Chrome & chromedriver versions,

Replace the toy classifier with a transformer-based classifier and provide training notebooks,

Add a Celery + Redis background worker for reliable async RPA execution,

Create a detailed README badge, CI (GitHub Actions) for tests and lint.
