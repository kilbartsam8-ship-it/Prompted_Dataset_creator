import os
import io
import re
import json
import zipfile
from dotenv import load_dotenv

# third-party
import pytesseract
from pdf2image import convert_from_bytes
from docx import Document
import PyPDF2

# LLM (keep as in your original)
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

OUTPUT_JSON = "output.json"

# -------------------------
# Helpers: file collection
# -------------------------
def folder_to_files(path: str, file_path: str | None = None) -> list:
    """
    Return a list of file paths to process.
    Accepts: directory path, zip path, or (optionally) a single file path.
    """
    files = []
    if file_path and os.path.isfile(file_path):
        files.append(os.path.abspath(file_path))
        return files

    if not path:
        return files

    path = os.path.abspath(path)
    if path.endswith(".zip") and os.path.isfile(path):
        extract_dir = os.path.join(os.path.dirname(path), "extracted_folder_for_resume_parser")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(extract_dir)
        for root, _, filenames in os.walk(extract_dir):
            for fn in filenames:
                files.append(os.path.join(root, fn))
        return files

    if os.path.isdir(path):
        for root, _, filenames in os.walk(path):
            for fn in filenames:
                files.append(os.path.join(root, fn))
        return files

    # single file
    if os.path.isfile(path):
        files.append(path)
    return files

# -------------------------
# Link classification
# -------------------------
def classify_links(links: list) -> dict:
    profile_patterns = {
        "LinkedIn": r"(linkedin\.com\/(in|company|pub)\/?|property\.linkedin\.com)",
        "GitHub": r"github\.com\/[A-Za-z0-9_.-]+",
        "Behance": r"behance\.net\/[A-Za-z0-9_.-]+",
        "Dribbble": r"dribbble\.com\/[A-Za-z0-9_.-]+",
        "Personal Website": r"^https?:\/\/[A-Za-z0-9.-]+\.(com|org|net|io|me|info|biz|ai)(\/.*)?$"
    }
    certificate_patterns = [
        r'certificates?\.edu',
        r'docsend\.com',
        r'publication\.org'
    ]

    classified = {"profile_links": {}, "certificate_links": [], "other_links": []}

    for link in sorted(set(links)):
        matched_profile = False
        for domain, pattern in profile_patterns.items():
            if re.search(pattern, link, re.I):
                # for multiple matches prefer more specific keys (keep first)
                if domain not in classified["profile_links"]:
                    classified["profile_links"][domain] = link
                matched_profile = True
                break
        if matched_profile:
            continue

        if any(re.search(p, link, re.I) for p in certificate_patterns):
            classified["certificate_links"].append(link)
        else:
            classified["other_links"].append(link)

    # dedupe & sort
    classified["certificate_links"] = sorted(set(classified["certificate_links"]))
    classified["other_links"] = sorted(set(classified["other_links"]))
    return classified

# -------------------------
# Project linking
# -------------------------
def assign_project_links(resume_text: str, project_sections: list, all_links: list) -> list:
    projects_with_links = []
    for project in project_sections:
        description = project.get("Description", "")
        project_links = [link for link in all_links if link in description]
        project["project_links"] = project_links
        projects_with_links.append(project)
    return projects_with_links

# -------------------------
# Extractors
# -------------------------
def extract_docx_text(file_bytes: bytes) -> dict:
    links = []
    try:
        doc = Document(io.BytesIO(file_bytes))
        texts = [p.text for p in doc.paragraphs]
        # hyperlinks extraction from relationships
        try:
            for rel in doc.part.rels.values():
                if "hyperlink" in rel.reltype:
                    # target_ref is the URL
                    links.append(rel.target_ref)
        except Exception:
            # some versions of python-docx store differently; ignore quietly
            pass
        return {"text": "\n".join(texts).strip(), "links": sorted(set(links))}
    except Exception as e:
        return {"text": f"[DOCX extraction failed: {e}]", "links": []}

def extract_text_from_textfile(file_bytes: bytes) -> dict:
    try:
        text = file_bytes.decode('utf-8', errors='ignore')
        return {"text": text.strip(), "links": []}
    except Exception as e:
        return {"text": f"[Text extraction failed: {e}]", "links": []}

def extract_pdf_text(file_bytes: bytes, ocr_if_needed: bool = True, max_ocr_pages: int = 8) -> dict:
    """
    Extract text from PDF using PyPDF2; if not enough text and ocr_if_needed True then run OCR.
    Returns dict: {text, links, used_ocr}
    """
    text_parts = []
    links = []
    used_ocr = False

    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            text_parts.append(page_text)

            # extract URI annotations if present
            try:
                annots = page.get("/Annots") or []
                if annots:
                    for annot in annots:
                        try:
                            obj = annot.get_object()
                            if "/A" in obj and "/URI" in obj["/A"]:
                                links.append(obj["/A"]["/URI"])
                            # some PDFs embed links under /A->/URI differently:
                            elif "/A" in obj and isinstance(obj["/A"], dict) and obj["/A"].get("/URI"):
                                links.append(obj["/A"].get("/URI"))
                        except Exception:
                            continue
            except Exception:
                # annotation parsing can fail depending on PDF structure
                pass

        joined = "\n".join(text_parts).strip()
    except Exception as e:
        joined = ""
        # keep going to OCR if allowed

    # If empty or suspiciously small, try OCR
    if (not joined or len(joined) < 200) and ocr_if_needed:
        try:
            images = convert_from_bytes(file_bytes)
            ocr_texts = []
            for i, img in enumerate(images):
                if i >= max_ocr_pages:
                    break
                ocr_texts.append(pytesseract.image_to_string(img))
            joined = "\n".join(ocr_texts).strip()
            used_ocr = True
        except Exception as e:
            # if OCR fails, put the error in the text
            if not joined:
                joined = f"[PDF extraction failed and OCR failed: {e}]"

    return {"text": joined[:8000], "links": sorted(set(links)), "used_ocr": used_ocr}

# -------------------------
# LLM / Groq interaction
# -------------------------
def llm_json_response(resume_text, structured_links):
    """
    Calls Groq LLM with the full system prompt (your provided prompt).
    Returns a dict with:
      - system_prompt: (the full system prompt string)
      - json_response: parsed JSON (dict/list) if possible, otherwise a string containing the JSON-like output or error info
    """
    # --- EXACT prompt you provided (kept verbatim) ---
    full_system_prompt = r"""
            "You are an AI bot parsing resumes with high accuracy (target 95%). "
            "First, determine if the resume belongs to a **Fresher** (no professional work experience beyond internships, "
            "recent graduation within 2 years, focus on education/projects) or an **Experienced professional** "
            "(mentions job roles, companies, and total experience > 1 year). Use these criteria strictly:\n"
            "- Fresher: Internship duration < 6 months, no full-time roles, recent education (e.g., 2023-2025).\n"
            "- Experienced: Full-time roles with total experience > 1 year, or senior roles.\n"
            "Return a JSON object with a mandatory 'classification' field ('Fresher' or 'Experienced') and extract the following fields based on the category. "
            "Ensure all fields are included, using empty strings ('') for unavailable data. Output only valid JSON, no additional text.\n\n"

            "Important Classification Refinement Rules:\n"
            "- If graduation year is within 2023–2025, assume the candidate is likely a Fresher unless there is clear, multi-year full-time work after graduation.\n"
            "- Treat 'Live Project', 'Internship', 'Academic Project', or 'Capstone Project' as internship experience — never as full-time work.\n"
            "- If the experience period overlaps with college years, treat it as internship or academic work, not full-time.\n"
            "- Do not assume the candidate is Experienced simply because there is a 'Work Experience' section.\n"
            "- If total full-time experience duration is less than 12 months, classify as 'Fresher'.\n"
            "- Prefer 'Fresher' classification when ambiguous — do not overestimate experience.\n"

            "Return a JSON object with a mandatory 'classification' field ('Fresher' or 'Experienced') and extract the following fields based on the category. "
            "Ensure all fields are included, using empty strings ('') for unavailable data. Output only valid JSON, no additional text.\n\n"

            "- For Experienced candidates, `experience.jobs` should **only include full-time jobs**."
            "- Internships must be listed **only in the `internships` field**, even if after graduation."
            "- Do not duplicate internships in `experience.jobs`."
            "- If the candidate has no full-time experience, set 'total_experience' to  null."
              "Do not guess or fill values."
            "- If any field is missing, return it as an empty string, empty list, or null."


            "Additionally, extract a dedicated field:\n"
            "'Technical skills' → a combined list of *every skill mentioned anywhere* "
            "(Skills section, Projects, Experience, Certifications, Summary, etc).\n\n"

            ... For fields like `Description` and `Responsibilities` in Experience, Projects, and Internships, 
                summarize them into 3–4 concise sentences that preserve all the important technical and contextual details. 
                Do not drop key technologies, roles, or outcomes. The goal is to condense long paragraphs into shorter 
                summaries without losing meaning.

            - For the 'Summary' field, condense the candidate’s profile summary into 3–5 sentences. 
                 Retain core skills, domain expertise, career goals, and unique strengths. 
                 Avoid generic filler text. Ensure the summary is professional, concise, and impactful.
            "If Fresher, extract:\n"
            "1. Name\n2. Email\n3. Phone number\n4. Profile links (LinkedIn, GitHub, etc)\n"
            "5. Education details (list of {Degree(include specialization if mentioned), College, Year, Grades})\n"
            "- Do not separate specialization into a new field. Include it in Degree.\n"
            "6. Technical skills (include skills mentioned in Skills section, job descriptions, project descriptions,Certifications and summary)\n"
            "7. Soft skills (list)\n8. Projects (list of {Title, Description,Technologies used, Responsibilities,project links if provided})\n"
            "9. Certifications (list)\n10. Languages known (list)\n11. Hobbies/extra-curriculars\n"
            "12. Current location\n13. Expected CTC\n14. Work/location preference\n15. work mode\n"
            "16. Internship details (list of {Company,Role,Period, Duration,Technologies used in each role, Responsibilities})\n17. Summary\n\n"

            "If Experienced, extract:\n"
            "1. Name\n2. Email\n3. Phone number\n4. Profile links (LinkedIn, GitHub, etc)\n"
            "5. Technical skills (include skills mentioned in Skills section, job descriptions, project descriptions,Certifications and summary)\n6. Soft skills (list)\n"
            "7. Total Experience\n"  
            "8. Experience details (list of {Company, Role,Period, Duration, Technologies used, Responsibilities})\n"
            "    - Only full-time professional roles. Do not include internships here.\n"
            "9. Internship details (list of {Company, Role,Period,Duration,Technologies used, Responsibilities})\n"
            "    - Only internships. Do not duplicate in Experience details.\n"
            "10. Projects (list of {Title, Description,Technologies used, Responsibilities,project links if provided})\n"
            "11. Certifications (list)\n"
            "12. Education details (list of {Degree(include specialization if mentioned), College, Year, Grades})\n"
            "   - Do not separate specialization into a new field. Include it in Degree.\n"
            "13. Current CTC\n14. Expected CTC\n"
            "15. Notice period\n16. Work mode\n17. Preferred location\n"
            "18. Summary\n19. Languages known (list)\n"
            "20. Hobbies/extra-curriculars\n"
            "21. Current location\n"

            "Return a JSON object with a mandatory 'classification' field and extract all fields as per instructions. give it in this format {{Base_Json}} with system_promptand entire json_response filled appropriately.\n\n"
            "Include profile_links, project_links, certificate_links, other_links from structured links. "
            "provide these links in the relevant sections (e.g., profile links in profile section, project links in projects to that particular project)."
            "Use empty string/list/null for missing data. Return valid JSON only.\n\n"
            """
    # --- end prompt ---

    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.3
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", full_system_prompt),
            ("system", f"Structured Links:\n{json.dumps(structured_links, indent=2)}"),
            ("user", "{resume_text}")
        ])
        chain = prompt | llm
        content = chain.invoke({"resume_text": resume_text})

        # First attempt: parse the whole output as JSON
        try:
            parsed = json.loads(content)
            return {"system_prompt": full_system_prompt, "json_response": parsed}
        except Exception:
            # Fallback: find first JSON-like substring and try parse
            json_match = re.search(r"(\{(?:[^{}]|(?R))*\})", content, re.S)
            if json_match:
                json_text = json_match.group(1)
                try:
                    parsed = json.loads(json_text)
                    return {"system_prompt": full_system_prompt, "json_response": parsed}
                except Exception:
                    # Return the JSON text as a string so your caller still receives something usable
                    return {"system_prompt": full_system_prompt, "json_response": json_text}
            # If nothing found, return raw content in json_response field (string)
            return {"system_prompt": full_system_prompt, "json_response": content}

    except Exception as e:
        # consistent failure shape
        return {"system_prompt": full_system_prompt, "json_response": json.dumps({"error": f"LLM call failed: {e}"})}

# -------------------------
# Main processing
# -------------------------
def extract_projects_from_text(text: str) -> list:
    """
    Extract simple Project blocks. Looks for 'Project:' lines, or 'Projects' section.
    Returns list of {"Title":..., "Description": ...}
    """
    projects = []
    # Project: Title \n description (until next Project or blank line)
    matches = re.findall(r"Project\s*[:\-]\s*(.+)\n(.*?)(?=\nProject\s*[:\-]|\n[A-Z][a-zA-Z\s]{1,30}?:|\Z)", text, re.S | re.I)
    for title, desc in matches:
        projects.append({"Title": title.strip(), "Description": re.sub(r"\n\s+", " ", desc.strip())})
    # If none found, try to find a Projects: section header
    if not projects:
        m = re.search(r"Projects\s*[:\n]\s*(.*?)(?=\n[A-Z][a-zA-Z\s]{1,30}?:|\Z)", text, re.S | re.I)
        if m:
            block = m.group(1).strip()
            # split by lines starting with - or *
            parts = re.split(r"\n-{2,}|\n\n+", block)
            for p in parts:
                line = p.strip()
                if not line:
                    continue
                # try to split Title - Description
                if "-" in line:
                    title, desc = line.split("-", 1)
                    projects.append({"Title": title.strip(), "Description": desc.strip()})
                else:
                    projects.append({"Title": line.split("\n", 1)[0][:80], "Description": line})
    return projects

def main(input_path: str):
    all_paths = folder_to_files(input_path)
    if not all_paths:
        print("No files found to process. Provide a folder, file or zip path.")
        return

    results = []
    processed = 0
    failed_files = []

    for fp in all_paths:
        filename = os.path.basename(fp)
        lower = filename.lower()
        meta = {"filename": filename, "ocr_used": False, "source_path": fp}

        try:
            with open(fp, "rb") as f:
                file_bytes = f.read()

            if lower.endswith(".pdf"):
                result = extract_pdf_text(file_bytes)
            elif lower.endswith(".docx"):
                result = extract_docx_text(file_bytes)
            elif lower.endswith(".txt"):
                result = extract_text_from_textfile(file_bytes)
            else:
                # unsupported file type: skip but record
                failed_files.append({"filename": filename, "reason": "Unsupported file type"})
                continue

            # classify links
            structured_links = classify_links(result.get("links", []))

            # detect projects and attach links
            project_sections = extract_projects_from_text(result.get("text", ""))
            projects_with_links = assign_project_links(result.get("text", ""), project_sections, result.get("links", []))
            structured_links["projects"] = projects_with_links

            # call LLM
            llm_result = llm_json_response(result.get("text", ""), structured_links)
            json_resp = llm_result.get("json_response")
            sys_prom = llm_result.get("system_prompt")

            # collect output object
            output_obj = {
                "messages": [
                    {"role":"system", "content": sys_prom,},
                    {"role":"user", "content": (result.get("text", "")[:300] + "...") if result.get("text") else "",},
                    {"role":"assistant", "content": json_resp}
                ]
            }

            results.append(output_obj)
            processed += 1

        except Exception as e:
            failed_files.append({"filename": filename, "error": str(e)})
            continue

    # write or append to output.json as array
    try:
        if os.path.exists(OUTPUT_JSON):
            # load and extend
            try:
                with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if isinstance(existing, list):
                    existing.extend(results)
                    out_list = existing
                else:
                    out_list = existing if existing else []
                    out_list = [out_list] + results
            except Exception:
                out_list = results
        else:
            out_list = results

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(out_list, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"Failed to write {OUTPUT_JSON}: {e}")

    print(f"Processed: {processed}, Failed: {len(failed_files)}, \nMeta: {meta}")
    if failed_files:
        print("Failed files detail:", json.dumps(failed_files, indent=2))

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    # change this to a folder path, single file, or zip file
    input_path = input("Enter your folder or file or zip path here : ")
    main(input_path)
