from fastapi import FastAPI, UploadFile, File, Request 
from fastapi.responses import HTMLResponse, JSONResponse 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from fastapi.templating import Jinja2Templates

import os


from app import (
    #extract_text_from_pdf_bytes,
    extract_text_from_scanned_pdf,
    structure_medical_data,
    build_patient_summary,
    retrieve_hedis_rules,
    analyze_care_gaps
)



app = FastAPI()

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict later in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Templates ----------
templates = Jinja2Templates(directory="templates")

# ---------- Static Files ----------
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------- HTML ROUTES ----------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/signup", response_class=HTMLResponse)
async def signup(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.get("/user_dashboard", response_class=HTMLResponse)
async def user(request: Request):
    return templates.TemplateResponse("user_dashboard.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/forgot_password", response_class=HTMLResponse)
async def forgot_password(request: Request):
    return templates.TemplateResponse("forgot_password.html", {"request": request})

# ----------- ANALYZE ENDPOINT ----------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # 1. Extract text using OCR only
        text = extract_text_from_scanned_pdf(contents)

        # 2. Structure patient
        patient_data = structure_medical_data(text)

        # 3. Retrieve HEDIS rules
        summary = build_patient_summary(patient_data)
        rules = retrieve_hedis_rules(summary)

        # 4. AI Care Gap Analysis
        result = analyze_care_gaps(patient_data, rules)

        return JSONResponse({"result": result})

    except Exception as e:
        return JSONResponse({"result": str(e)}, status_code=500)
