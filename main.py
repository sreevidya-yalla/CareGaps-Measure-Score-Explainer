from fastapi import FastAPI, UploadFile, File, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer

from jose import JWTError, jwt
from datetime import datetime, timedelta
import secrets
import smtplib
from email.message import EmailMessage
import os
import uuid
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from dotenv import load_dotenv


from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHashError

# Import AI functions
from app import (
    extract_text_from_scanned_pdf,
    extract_text_from_pdf,
    structure_medical_data,
    build_patient_summary,
    retrieve_hedis_rules,
    analyze_care_gaps
)

load_dotenv()

app = FastAPI()



# DATABASE
# ==============================
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI not found in .env file")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["caregap_db"]
users_collection = db["users"]

# Create unique index on email
try:
    users_collection.create_index("email", unique=True)
except Exception:
    pass  


# PASSWORD HASHING 
# ==============================

ph = PasswordHasher(
    time_cost=3,           # 3 iterations
    memory_cost=65536,     # 64 MB memory
    parallelism=4,         # 4 threads
    hash_len=32,           # 32 byte output
    salt_len=16            # 16 byte salt
)

def hash_password(password: str) -> str:
    """
    Hash a password using Argon2id
    Handles passwords of ANY length (even 10,000+ characters)
    """
    try:
        # Argon2 automatically handles long passwords
        return ph.hash(password)
    except Exception as e:
        print(f"Password hashing error: {e}")
        raise HTTPException(status_code=500, detail="Error processing password")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its Argon2 hash
    """
    try:
        ph.verify(hashed_password, plain_password)
        return True
    except (VerifyMismatchError, InvalidHashError):
        return False
    except Exception as e:
        print(f"Password verification error: {e}")
        return False

def password_needs_rehash(hashed_password: str) -> bool:
    """
    Check if a password hash needs to be updated with current parameters
    Useful for upgrading security over time
    """
    try:
        return ph.check_needs_rehash(hashed_password)
    except:
        return False


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


# FILE UPLOAD CONFIG
# ==============================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# CORS
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TEMPLATES & STATIC
# ==============================
templates = Jinja2Templates(directory="templates")

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


    
# EMAIL FUNCTION
# ==============================
def send_reset_email(to_email, reset_link):
    msg = EmailMessage()
    msg["Subject"] = "Password Reset - HEDIS Explainer"
    msg["From"] = os.getenv("EMAIL_FROM")
    msg["To"] = to_email

    msg.set_content(f"""
Click the link below to reset your password:

{reset_link}

This link expires in 15 minutes.
""")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(os.getenv("EMAIL_USERNAME"), os.getenv("EMAIL_PASSWORD"))
            smtp.send_message(msg)
    except Exception as e:
        print("EMAIL FAILED:")
        print(e)
        



# AUTH ROUTES
# ==============================
@app.post("/api/signup")
async def register_user(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form("user"),
    admin_token: str = Form(None)  
):
    try:
        # Check if user already exists
        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")

        # ADMIN VALIDATION
        if role == "admin":
            expected_token = os.getenv("ADMIN_SIGNUP_TOKEN")

            if not expected_token:
                raise HTTPException(status_code=500, detail="Admin token not configured")

            if not admin_token or admin_token != expected_token:
                raise HTTPException(status_code=403, detail="Invalid admin signup token")

        # Hash password
        hashed_password = hash_password(password)

        users_collection.insert_one({
            "username": username,
            "email": email,
            "password": hashed_password,
            "role": role,
            "created_at": datetime.utcnow()
        })

        return {"message": "User registered successfully"}

    except HTTPException:
        raise
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="User already exists")
    except Exception as e:
        print(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")
    

    
@app.post("/api/login")
async def login_user(
    email: str = Form(...),
    password: str = Form(...)
):
    try:
        user = users_collection.find_one({"email": email})

        # Verify credentials
        if not user or not verify_password(password, user["password"]):
            raise HTTPException(status_code=400, detail="Invalid email or password")


        if password_needs_rehash(user["password"]):
            new_hash = hash_password(password)
            users_collection.update_one(
                {"_id": user["_id"]},
                {"$set": {"password": new_hash}}
            )

        # Create access token
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
    except HTTPException:
        raise
    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/api/logout")
async def logout():
    return {"message": "Logged out successfully"}

@app.post("/api/forgot-password")
async def forgot_password(email: str = Form(...)):
    try:
        user = users_collection.find_one({"email": email})

        if not user:
            return {"message": "If this email exists, a reset link has been sent."}

        reset_token = secrets.token_urlsafe(32)

        users_collection.update_one(
            {"email": email},
            {
                "$set": {
                    "reset_token": reset_token,
                    "reset_token_expiry": datetime.utcnow() + timedelta(minutes=15)
                }
            }
        )

        #reset_link = f"http://localhost:8000/reset-password/{reset_token}"
        BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

        reset_link = f"{BASE_URL}/reset-password/{reset_token}"
        send_reset_email(email, reset_link)

        return {"message": "If this email exists, a reset link has been sent."}
    except Exception as e:
        print(f"Forgot password error: {e}")
        return {"message": "If this email exists, a reset link has been sent."}

@app.post("/api/reset-password/{token}")
async def reset_password(
    token: str,
    new_password: str = Form(...)
):
    try:
        user = users_collection.find_one({
            "reset_token": token,
            "reset_token_expiry": {"$gt": datetime.utcnow()}
        })

        if not user:
            raise HTTPException(status_code=400, detail="Invalid or expired token")

        hashed_password = hash_password(new_password)

        users_collection.update_one(
            {"_id": user["_id"]},
            {
                "$set": {"password": hashed_password},
                "$unset": {"reset_token": "", "reset_token_expiry": ""}
            }
        )

        return {"message": "Password reset successful"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Reset password error: {e}")
        raise HTTPException(status_code=500, detail="Password reset failed")



# ADMIN USER MANAGEMENT APIs
# ==============================

@app.get("/admin/api/users")
async def get_users(
    page: int = 1,
    limit: int = 10,
    search: str = "",
    current_user: dict = Depends(get_current_admin)
):
    skip = (page - 1) * limit

    query = {}

    if search:
        query = {
            "$or": [
                {"username": {"$regex": search, "$options": "i"}},
                {"email": {"$regex": search, "$options": "i"}}
            ]
        }

    total_users = users_collection.count_documents(query)
    total_pages = (total_users + limit - 1) // limit

    users = list(
        users_collection.find(query, {"password": 0})
        .sort("created_at", -1)
        .skip(skip)
        .limit(limit)
    )

    # Convert ObjectId to string
    for user in users:
        user["_id"] = str(user["_id"])

    return {
        "total": total_users,
        "pages": total_pages,
        "users": users
    }






    
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
async def forgot_password_page(request: Request):
    return templates.TemplateResponse("forgot_password.html", {"request": request})

@app.get("/reset-password/{token}", response_class=HTMLResponse)
async def reset_password_page(request: Request, token: str):
    return templates.TemplateResponse(
        "reset_password.html",
        {"request": request, "token": token}
    )


#
#
#

@app.get("/debug-email")
async def debug_email(current_user: dict = Depends(get_current_admin)):
    return {
        "EMAIL_FROM": os.getenv("EMAIL_FROM"),
        "EMAIL_USERNAME": os.getenv("EMAIL_USERNAME"),
        "EMAIL_PASSWORD_EXISTS": bool(os.getenv("EMAIL_PASSWORD"))
    }






    
# PROTECTED ANALYZE ROUTE
# ==============================
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        contents = await file.read()
        
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        file_id = str(uuid.uuid4())
        file_path = f"{UPLOAD_DIR}/{file_id}.pdf"

        with open(file_path, "wb") as f:
            f.write(contents)

        # Extract text from PDF
        text = extract_text_from_scanned_pdf(contents)
        
        if not text:
            with open(file_path, "rb") as f:
                text = extract_text_from_pdf(f)

        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        # Processing the text
        patient_data = structure_medical_data(text)
        
        if "error" in patient_data:
            raise HTTPException(status_code=500, detail="Failed to structure medical data")

        summary = build_patient_summary(patient_data)
        rules = retrieve_hedis_rules(summary)
        result = analyze_care_gaps(patient_data, rules)

        return {
            "result": result,
            "file_url": f"/uploads/{file_id}.pdf",
            "analyzed_by": current_user["sub"]
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
