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
import os
from dotenv import load_dotenv

load_dotenv()


# ---------------- CONFIG ----------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

client = genai.Client(api_key=api_key)


embed_model = SentenceTransformer('all-MiniLM-L6-v2')

index = faiss.read_index("hedis_index.faiss")

with open("hedis_chunks.pkl", "rb") as f:
    hedis_chunks = pickle.load(f)

# ---------------- HEDIS RETRIEVAL ----------------
def retrieve_hedis_rules(query, k=3):
    query_vec = embed_model.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [hedis_chunks[i] for i in I[0]]


# ---------------- PDF TEXT EXTRACTION ----------------
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except:
        return ""
    return text.strip()

# ---------------- OCR FROM IMAGE ----------------
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# ---------------- OCR FROM SCANNED PDF ----------------
def extract_text_from_scanned_pdf(file):
    images = convert_from_bytes(file)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img) + "\n"
    return text.strip()

# ---------------- STRUCTURE MEDICAL DATA ----------------
def structure_medical_data(raw_text):
    prompt = f"""
    Extract structured medical information from this patient report.

    Return ONLY valid JSON with:
    age, gender,
    conditions (list),
    lab_results (list of {{test, value, date}}),
    vitals (list of {{type, value, date}}),
    screenings (list of {{test, date}})

    TEXT:
    {raw_text}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    raw_output = response.text.strip()

    #  Remove markdown ```json ``` if present
    if raw_output.startswith("```"):
        raw_output = raw_output.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw_output)
    except Exception as e:
        return {"error": "Could not parse structured data", "raw": raw_output}


# ---------------- DATE PARSER ----------------
def parse_date(date_str):
    try:
        return parser.parse(date_str)
    except:
        return None
    
# ---------------- PATIENT SUMMARY ----------------
def build_patient_summary(patient):
    conditions = ", ".join(patient.get("conditions", []))
    return f"{patient.get('age')} year old {patient.get('gender')} with {conditions}"


# ---------------- CARE GAP ANALYSIS ----------------
def analyze_care_gaps(patient_data, hedis_rules):
    prompt = f"""
    
You are a healthcare quality assistant.

GOAL:
Identify preventive care gaps using ONLY the provided HEDIS guidelines and the patient data.

OUTPUT FORMAT:
1. Care Gaps Found (max 5 bullet points)
2. Why It Matters (short, simple explanation)
3. Suggested General Preventive Actions (safe, non-medical)
4. Confidence Note (1 short sentence)

RULES:
- Do NOT diagnose diseases.
- Do NOT suggest medications or dosages.
- Do NOT invent information not present in the patient data or guidelines.
- Use simple, clear language suitable for a non-medical patient.
- Use soft language like "not clearly documented" instead of "missing".
- Limit to the most important gaps only.
- If no care gaps are found, clearly state: "No major care gaps found."
- Use round dot bullets (•) for every point but not for headings.
- Do NOT use *, - . 

STYLE GUIDELINES:
- Use natural, conversational language.
- Avoid repeating the same sentence starters.
- Vary verbs like discuss, consider, review, check.
- Sound like a helpful health assistant, not a technical report.
- Keep it simple but human.


PATIENT DATA:
{patient_data}

HEDIS GUIDELINES:
{hedis_rules}
"""



    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text





