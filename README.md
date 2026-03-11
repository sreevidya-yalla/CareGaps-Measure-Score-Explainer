# CareGap Capstone Project: HEDIS Explainer

## Overview
The **CareGap Capstone Project** is a web-based application built with **FastAPI** that analyzes patient medical records and identifies preventive care gaps based on **HEDIS** (Healthcare Effectiveness Data and Information Set) guidelines. 

The system takes patient documents (scanned or native PDFs), extracts text using **Tesseract OCR** and `pdfplumber`, structures the data using the **Google Gemini AI**, and matches it against HEDIS rules stored in a local **FAISS** vector database. Finally, it uses Gemini AI to output patient-friendly care gap analysis and preventive action suggestions.

## Key Features
- **PDF Processing**: Extracts text from both regular and scanned medical records using `pdfplumber` and `Tesseract OCR`.
- **Medical Data Structuring**: Uses Google Gemini AI to extract structured medical data (age, gender, conditions, lab results, vitals, screenings) into JSON format.
- **HEDIS Guidelines Integration**: Retrieves relevant HEDIS rules using a local FAISS vector database.
- **Care Gap Analysis**: Analyzes patient data against HEDIS rules to identify care gaps and suggests preventive actions using Gemini AI in plain, non-medical language.
- **Secure Architecture**: 
  - User registration, login, and robust password reset using Argon2 hashing.
  - JWT tokens for securing endpoints.
  - Backend integration with MongoDB via PyMongo.
- **Admin Dashboard**: Admin panel for managing and tracking users.

## Project Structure
- `main.py`: The entry point for the FastAPI server. It contains authentication routes mapping, UI rendering, database integration, and the main `/analyze` endpoint securely interacting with AI modules.
- `app.py`: Contains modular AI functions:
  - Text extraction natively and via OCR mappings.
  - Gemini AI prompts for structuring data.
  - Sentence transformers for embedding and searching the FAISS index.
  - Care gap NLP interaction with Gemini based on retrieved rules.
- `chunking.py`: Utility script to process and chunk raw HEDIS PDF guidelines into vectors, feeding them into the FAISS index `hedis_index.faiss`.
- `requirements.txt`: Python dependencies required for the project.
- `templates/` & `static/`: Contains HTML layout templates and static assets for the application frontend pages.
- `uploads/`: Temporary directory for uploaded user PDFs.

## Prerequisites
Ensure the following tools and services are installed:
- **Python 3.8+**
- **MongoDB**: A local instance or MongoDB Atlas.
- **Tesseract OCR**: Required for text extraction from scanned documents.
- **Poppler**: Required for converting PDF files to images (for OCR processing).
- **Google Gemini API Key**: Needed to use GenAI LLMs.

## Setup Instructions

### 1. Clone & Dependencies
Clone the repository and install the required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root directory (where `main.py` is present) and configure the following variables:
```
# Gemini AI
GEMINI_API_KEY=your_gemini_api_key_here

# MongoDB Setup
MONGO_URI=mongodb_connection_string

# Authorization
SECRET_KEY=your_jwt_secret_key_here

# Admin Signup
ADMIN_SIGNUP_TOKEN=secure_token_for_admin_signup

# Email Configurations for Password Reset
EMAIL_FROM=your_sender_email
EMAIL_USERNAME=your_sender_email
EMAIL_PASSWORD=your_email_app_password
BASE_URL=http://localhost:8000
```

### 3. Tesseract and Poppler configuration
Adjust the absolute paths within `app.py` based on your system installation. 
- `TESSERACT_PATH` is initialized for `Tesseract-OCR/tesseract.exe`
- `POPPLER_PATH` is initialized for `poppler/Library/bin`
*(The project attempts local fallback or system paths if they are available).*

### 4. Vector Database Generation
If you do not have `hedis_index.faiss` and `hedis_chunks.pkl` in your root folder, generate them by running `chunking.py` against your HEDIS PDF descriptions:
```bash
python chunking.py
```

### 5. Running the Application
Start the uvicorn server in development mode:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Alternatively, as it is structured:
```bash
python main.py
```
> The web application will serve traffic on http://localhost:8000

## API Endpoints Overview
### Authentication
- `POST /api/signup`: Register a new user (`role` can be `admin` requiring an admin token).
- `POST /api/login`: Returns an entry JWT securely handling auth.
- `POST /api/logout`: Simple logout mechanism.
- `POST /api/forgot-password` & `POST /api/reset-password/{token}`: Password management endpoints.

### Application Logic
- `POST /analyze`: Extracts facts from the PDF (sent within form-data) and outputs HEDIS-compliant care-gap points.

### Pages & Dashboards
- `GET /`: General presentation or login entry.
- `GET /user_dashboard`: The interface for general practitioners or patients to upload their PDF.
- `GET /admin`: Dashboard accessible only by users bearing the admin token, mapping over `/admin/api/users`.

## Technologies Used
- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **LLM/AI Model**: Google Gemini (via `google-genai`)
- **Embeddings & Vector Search**: `Sentence-Transformers`, [FAISS](https://faiss.ai/)
- **OCR Engine**: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) + [Poppler](https://poppler.freedesktop.org/)
- **Database**: [MongoDB](https://www.mongodb.com/)
- **Security**: Argon2, PyJWT
