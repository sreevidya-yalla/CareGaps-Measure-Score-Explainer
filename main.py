from fastapi import FastAPI, UploadFile, File, Request, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer

from jose import JWTError, jwt
from datetime import datetime, timedelta

import os
import uuid
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# ==============================
# DATABASE
# ==============================
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI not found in .env file")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["caregap_db"]
users_collection = db["users"]

users_collection.create_index("email", unique=True)

# ==============================
# PASSWORD HASHING
# ==============================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ==============================
# JWT CONFIG
# ==============================
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY not set in .env")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def get_current_admin(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


# ==============================
# IMPORT AI FUNCTIONS
# ==============================
from app import (
    extract_text_from_scanned_pdf,
    structure_medical_data,
    build_patient_summary,
    retrieve_hedis_rules,
    analyze_care_gaps
)

# ==============================
# FILE UPLOAD CONFIG
# ==============================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==============================
# CORS
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# TEMPLATES & STATIC
# ==============================
templates = Jinja2Templates(directory="templates")

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ==============================
# AUTH ROUTES
# ==============================

@app.post("/api/signup")
async def register_user(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form("user")
):
    try:
        hashed_password = pwd_context.hash(password)

        users_collection.insert_one({
            "username": username,
            "email": email,
            "password": hashed_password,
            "role": role,
            "created_at": datetime.utcnow()
        })

        return {"message": "User registered successfully"}

    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="User already exists")


@app.post("/api/login")
async def login_user(
    email: str = Form(...),
    password: str = Form(...)
):
    user = users_collection.find_one({"email": email})

    if not user or not pwd_context.verify(password, user["password"]):
        raise HTTPException(status_code=400, detail="Invalid email or password")

    access_token = create_access_token(
        data={
            "sub": email,
            "role": user.get("role", "user")
        }
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": user.get("role", "user")
    }


@app.post("/api/logout")
async def logout():
    return {"message": "Logged out successfully"}


# ==============================
# HTML ROUTES
# ==============================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@app.get("/user_dashboard", response_class=HTMLResponse)
async def user_dashboard(request: Request):
    return templates.TemplateResponse("user_dashboard.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})


@app.get("/forgot_password", response_class=HTMLResponse)
async def forgot_password(request: Request):
    return templates.TemplateResponse("forgot_password.html", {"request": request})


# ==============================
# PROTECTED ANALYZE ROUTE
# ==============================

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        contents = await file.read()

        file_id = str(uuid.uuid4())
        file_path = f"{UPLOAD_DIR}/{file_id}.pdf"

        with open(file_path, "wb") as f:
            f.write(contents)

        text = extract_text_from_scanned_pdf(contents)
        patient_data = structure_medical_data(text)
        summary = build_patient_summary(patient_data)
        rules = retrieve_hedis_rules(summary)
        result = analyze_care_gaps(patient_data, rules)

        return {
            "result": result,
            "file_url": f"/uploads/{file_id}.pdf",
            "analyzed_by": current_user["sub"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
