import os
import sys
import pdfplumber
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
from google import genai
import json
from datetime import datetime
from dateutil import parser
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


# CONFIGURATION WITH EXACT PATHS
# ==============================

# Gets the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


TESSERACT_PATH = os.path.join(BASE_DIR, "Tesseract-OCR", "tesseract.exe")
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    print(f"✅ Tesseract found at: {TESSERACT_PATH}")
else:
    print(f"⚠️ Tesseract not found at: {TESSERACT_PATH}")
    
    alt_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for alt_path in alt_paths:
        if os.path.exists(alt_path):
            pytesseract.pytesseract.tesseract_cmd = alt_path
            print(f"✅ Tesseract found at fallback: {alt_path}")
            break


POPPLER_PATH = os.path.join(BASE_DIR, "poppler", "Library", "bin")
if os.path.exists(POPPLER_PATH):
    
    os.environ["PATH"] = POPPLER_PATH + os.pathsep + os.environ.get("PATH", "")
    print(f"✅ Poppler found at: {POPPLER_PATH}")
    
    # Verify pdftoppm exists
    pdftoppm_path = os.path.join(POPPLER_PATH, "pdftoppm.exe")
    if os.path.exists(pdftoppm_path):
        print(f"✅ pdftoppm.exe found")
    else:
        print(f"⚠️ pdftoppm.exe not found in {POPPLER_PATH}")
else:
    print(f"⚠️ Poppler not found at: {POPPLER_PATH}")

# Gemini API setup
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

client = genai.Client(api_key=api_key)

# Embedding model for HEDIS retrieval
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Loading FAISS index and chunks
try:
    index_path = os.path.join(BASE_DIR, "hedis_index.faiss")
    chunks_path = os.path.join(BASE_DIR, "hedis_chunks.pkl")
    
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            hedis_chunks = pickle.load(f)
        print(f" Loaded {len(hedis_chunks)} HEDIS chunks")
    else:
        print(" HEDIS index files not found. Create them first.")
        index = None
        hedis_chunks = []
except Exception as e:
    print(f" Error loading HEDIS files: {e}")
    index = None
    hedis_chunks = []

    
# PDF TEXT EXTRACTION
# ==============================

def extract_text_from_pdf(file):
    """
    Extract text from a regular (non-scanned) PDF
    """
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text.strip()

def extract_text_from_scanned_pdf(file_bytes):
    """
    Extract text from a scanned PDF using OCR
    Using your specific Poppler and Tesseract paths
    """
    try:
        if not file_bytes:
            return ""
        
        print(f"Converting PDF to images using Poppler...")
        
        # Converting PDF to images using your Poppler path
        if os.path.exists(POPPLER_PATH):
            images = convert_from_bytes(file_bytes, poppler_path=POPPLER_PATH)
        else:
            print("Using system Poppler as fallback...")
            images = convert_from_bytes(file_bytes)
        
        print(f"✅ Converted {len(images)} pages to images")
        
        text = ""
        for i, img in enumerate(images):
            print(f"Processing page {i+1}/{len(images)} with Tesseract OCR...")
            page_text = pytesseract.image_to_string(img)
            if page_text:
                text += page_text + "\n"
                print(f"  Extracted {len(page_text)} characters")
        
        return text.strip()
    except Exception as e:
        print(f"Error processing scanned PDF: {e}")
        return ""  


# STRUCTURE MEDICAL DATA
# ==============================
def structure_medical_data(raw_text):
    """
    Use Gemini to extract structured medical data from raw text
    """
    if not raw_text or len(raw_text.strip()) == 0:
        return {"error": "No text to analyze", "raw": ""}
    
    print(f"Structuring medical data from {len(raw_text)} characters...")
    
    prompt = f"""
    Extract structured medical information from this patient report.
    
    Return ONLY valid JSON with these fields:
    - age (number)
    - gender (string)
    - conditions (list of strings)
    - lab_results (list of objects with fields: test, value, date)
    - vitals (list of objects with fields: type, value, date)
    - screenings (list of objects with fields: test, date)
    
    If a field is not found, use empty list or null.
    
    TEXT:
    {raw_text[:3000]}  # Limit text to avoid token limits
    """

    raw_output = ""  
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        raw_output = response.text.strip()

        if raw_output.startswith("```"):
            raw_output = raw_output.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(raw_output)
        print("✅ Successfully structured medical data")
        return parsed
    except Exception as e:
        print(f"Error structuring medical data: {e}")
        return {
            "error": f"Could not parse structured data: {str(e)}", 
            "raw": raw_output if raw_output else "No output from Gemini"
        }


# DATE PARSER
# ==============================
def parse_date(date_str):
    """
    Parse date string to datetime object
    """
    try:
        return parser.parse(date_str)
    except:
        return None




# PATIENT SUMMARY
# ==============================
def build_patient_summary(patient):
    """
    Create a human-readable patient summary
    """
    if "error" in patient:
        return "Patient data not available"
    
    conditions = ", ".join(patient.get("conditions", []))
    age = patient.get('age', 'Unknown age')
    gender = patient.get('gender', 'Unknown gender')
    
    if conditions:
        return f"{age} year old {gender} with {conditions}"
    else:
        return f"{age} year old {gender}"


    
# HEDIS RETRIEVAL
# ==============================
def retrieve_hedis_rules(query, k=3):
    """
    Retrieve relevant HEDIS rules based on the query
    """
    if index is None or not hedis_chunks:
        return ["HEDIS guidelines not available. Please upload guidelines first."]
    
    query_vec = embed_model.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [hedis_chunks[i] for i in I[0]]


    
# CARE GAP ANALYSIS
# ==============================
def analyze_care_gaps(patient_data, hedis_rules):
    """
    Analyze care gaps using patient data and HEDIS guidelines
    """
    if "error" in patient_data:
        return f"Error in patient data: {patient_data['error']}"
    
    prompt = f"""
    You are a healthcare quality assistant.
    
    GOAL:
    Identify preventive care gaps using the provided HEDIS guidelines and the patient data.
    
    OUTPUT FORMAT:
    1. Care Gaps Found (max 5 bullet points)
    2. Why It Matters (short, simple explanation)
    3. Suggested General Preventive Actions (safe, non-medical)
    
    RULES:
    - Do NOT diagnose diseases.
    - Do NOT suggest medications or dosages.
    - Do NOT invent information not present in the patient data or guidelines.
    - Use simple, clear language suitable for a non-medical patient.
    - Use soft language like "not clearly documented" instead of "missing".
    - Limit to the most important gaps only.
    - If no care gaps are found, clearly state: "No major care gaps found."
    - Use round dot bullets (•) for every point but not for headings.
    - DO NOT use *, - . 
    
    STYLE GUIDELINES:
    - Use natural, conversational language.
    - Avoid repeating the same sentence starters.
    - Vary verbs like discuss, consider, review, check.
    - Sound like a helpful health assistant, not a technical report.
    - Keep it simple but human.
    
    PATIENT DATA:
    {json.dumps(patient_data, indent=2)}
    
    HEDIS GUIDELINES:
    {hedis_rules}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error analyzing care gaps: {str(e)}"



# TEST FUNCTION
# ==============================
def test_paths():
    """Test if all paths are configured correctly"""
    print("\n" + "="*60)
    print("TESTING PATH CONFIGURATION")
    print("="*60)
    
    print(f"\n📁 Project Base Directory: {BASE_DIR}")
    
    # Test Tesseract
    print(f"\n🔍 Testing Tesseract:")
    print(f"   Path: {TESSERACT_PATH}")
    print(f"   Exists: {os.path.exists(TESSERACT_PATH)}")
    if os.path.exists(TESSERACT_PATH):
        try:
            version = pytesseract.get_tesseract_version()
            print(f"   Version: {version}")
            print(f"   ✅ Tesseract is working!")
        except Exception as e:
            print(f"   ❌ Tesseract error: {e}")
    
    # Test Poppler
    print(f"\n🔍 Testing Poppler:")
    print(f"   Path: {POPPLER_PATH}")
    print(f"   Exists: {os.path.exists(POPPLER_PATH)}")
    if os.path.exists(POPPLER_PATH):
        pdftoppm = os.path.join(POPPLER_PATH, "pdftoppm.exe")
        print(f"   pdftoppm.exe exists: {os.path.exists(pdftoppm)}")
        if os.path.exists(pdftoppm):
            print(f"   ✅ Poppler is working!")
    
    # Test HEDIS files
    print(f"\n🔍 Testing HEDIS files:")
    index_path = os.path.join(BASE_DIR, "hedis_index.faiss")
    chunks_path = os.path.join(BASE_DIR, "hedis_chunks.pkl")
    print(f"   Index exists: {os.path.exists(index_path)}")
    print(f"   Chunks exists: {os.path.exists(chunks_path)}")
    
    print("\n" + "="*60)
    if (os.path.exists(TESSERACT_PATH) and 
        os.path.exists(os.path.join(POPPLER_PATH, "pdftoppm.exe"))):
        print("✅ All paths configured correctly! Ready to run.")
    else:
        print("⚠️ Some paths are missing. Check the output above.")
    print("="*60)

# Run test if this file is executed directly
if __name__ == "__main__":
    test_paths()
