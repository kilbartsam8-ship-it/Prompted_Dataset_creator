import json
import PyPDF2
from docx import Document
import re
import pytesseract
from pdf2image import convert_from_bytes
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import io
import zipfile

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

Json_file = "output.json"

Base_Json = {
    "system_prompt" : "",
    "json_response" : {}
    }

# conversion of folder to list of files
def folder_to_files(folder_path, file_path=None):
    files = []
    try:
        if folder_path.endswith('.zip'):
            with zipfile.ZipFile(folder_path, 'r') as zip_ref:
                zip_ref.extractall('extracted_folder')
            for root, _, filenames in os.walk('extracted_folder'):
                for filename in filenames:
                    files.append(os.path.join(root, filename))

        elif os.path.isdir(folder_path):
            for root, _, filenames in os.walk(folder_path):
                for filename in filenames:
                    files.append(os.path.join(root, filename))
                
        elif file_path and os.path.isfile(file_path):
            files.append(file_path)
            
        return files

    except Exception as e:
        print(f"Error accessing folder: {e}")
        return files 

def classify_links(links):
    profile_patterns = {
        "LinkedIn" : r'propertys\.linkedin\.com|linkedin\.com\/in\/|linkedin\.com\/company\/',
        "GitHub" : r'github\.com\/[A-Za-z0-9_-]+|github\.com$',
        "Behance" : r'behance\.net\/[A-Za-z0-9_-]+',
        "Dribbble" : r'dribbble\.com\/[A-Za-z0-9_-]+',
        "Personal Website" : r'^[a-zA-Z0-9.-]+\.(com|org|net|io|me|info|biz|ai)(\/)?$',
        "Online Portfolio" : r'^[a-zA-Z0-9.-]+\.(com|org|net|io|me|info|biz|ai)(\/)?$'
    }
    certificate_patterns = [
        r'certificates\.edu',
        r'docsend\.com',
        r'publication\.org'
    ]

    classified = {"profile_links": {}, "certificate_links": [], "other_links": []}

    for link in links:
        matched = False
        for domain, pattern in profile_patterns.items():
            if re.search(pattern, link, re.I):
                classified["profile_links"][domain] = link
                matched = True
                break
            if not matched:
                if any(re.search(p, link, re.I) for  p in certificate_patterns):
                    classified["certificate_links"].append(link)
                else:
                    classified["other_links"].append(link)
    classified["certificate_links"] = sorted(set(classified["certificate_links"]))
    classified["other_links"] = sorted(set(classified["other_links"]))
    return classified

def assign_project_links(resume_text, project_sections, all_links):
    projects_with_links = []
    for project in project_sections:
        description = project.get("Description", "")
        project_links = [link for link in all_links if link in description]
        project["project_links"] = project_links
        projects_with_links.append(project)
    return projects_with_links

def extract_docx_text(file_bytes: bytes) -> str:
    links = []
    try:
        doc = Document(io.BytesIO(file_bytes))
        texts = [p.text for p in doc.paragraphs]
        for rel in doc.part.rels.values():
            if "hyperlink" in rel.reltype:
                links.append(rel.target_ref)
        return {"text": "\n".join(texts).strip(), "links": sorted(set(links))}

    except Exception as e:
        return f"[DOCX extraction failed: {e}]"

def extract_pdf_text(file_bytes: bytes, ocr_if_needed=True) -> tuple:
    used_ocr = False
    text_parts = []
    links = []
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
        joined = "\n".join(text_parts).strip()
        if "/Annots" in page:
            for annot in page["/Annots"]:
                obj = annot.get_object()
                if "/A" in obj and "/URI" in obj["/A"]:
                    links.append(obj["/A"]["/URI"])
        return {"text": joined, "links": sorted(set(links))}
    except Exception:
        joined = ""

    if (not joined or len(joined) < 200) and ocr_if_needed:
        try:
            images = convert_from_bytes(file_bytes)
            ocr_text_parts = []
            for img in images:
                ocr_text_parts.append(pytesseract.image_to_string(img))
            joined = "\n".join(ocr_text_parts)
            used_ocr = True
        except Exception as e:
            joined = joined or f"[PDF extraction failed and OCR failed: {e}]"
    return {"text": joined[:8000], "links": sorted(set(links)), "used_ocr": used_ocr}

def extract_text_from_textfile(file_bytes: bytes) -> str:
    links = []
    try:
        text = file_bytes.decode('utf-8', errors='ignore')
        return {"text": text.strip(), "links": sorted(set(links))}
    except Exception as e:
        return f"[Text file extraction failed: {e}]"

def llm_json_response(resume_text, structured_links):
        
    try:
        llm = ChatGroq(
                groq_api_key=api_key,
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
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

        prompt = ChatPromptTemplate.from_messages([
            ("system", full_system_prompt),
            ("system", f"Structured Links:\n{json.dumps(structured_links, indent=2)}"),
            ("user", "{resume_text}")
            ])
        chain = prompt | llm
        content = chain.invoke({"resume_text": resume_text})

        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            return json_match.group(0)
        
    except Exception as e:
        return json.dumps({"classification": "unknown", "error": "Invalid JSON returned"})
    


def main():
    extracted_files = folder_to_files("Enter your folder or file or zip path here")
    file_count = 0
    file_not_processed = []
    try:
        for file in extracted_files:
        
            filename = file.name
            file_bytes = file.read()
            meta = {"filename": filename, "ocr_used": False}
            lower = filename.lower()
            if lower.endswith('.pdf'):
                result = extract_pdf_text(file_bytes)
                structured_links = classify_links(result["links"])
                project_sections = []
                project_matches = re.findall(r"(Project\s*:\s*(.+?)\n(.*?)(?=\nProject|$))", result["text"], re.S | re.I)
                for _, title, desc in project_matches:
                    project_sections.append({"Title": title.strip(), "Description": desc.strip()})
                projects_with_links = assign_project_links(result["text"], project_sections, result["links"])
                structured_links["projects"] = projects_with_links
                meta["ocr_used"] = result.get("used_ocr", False)
            elif lower.endswith('.docx'):
                result = extract_docx_text(file_bytes)
                structured_links = classify_links(result["links"])
                project_sections = []
                project_matches = re.findall(r"(Project\s*:\s*(.+?)\n(.*?)(?=\nProject|$))", result["text"], re.S | re.I)
                for _, title, desc in project_matches:
                    project_sections.append({"Title": title.strip(), "Description": desc.strip()})
                projects_with_links = assign_project_links(result["text"], project_sections, result["links"])
                structured_links["projects"] = projects_with_links
            elif lower.endswith('.txt'):
                result = extract_text_from_textfile(file_bytes)
                structured_links = classify_links(result["links"])
                project_sections = []
                project_matches = re.findall(r"(Project\s*:\s*(.+?)\n(.*?)(?=\nProject|$))", result["text"], re.S | re.I)
                for _, title, desc in project_matches:
                    project_sections.append({"Title": title.strip(), "Description": desc.strip()})
                projects_with_links = assign_project_links(result["text"], project_sections, result["links"])
                structured_links["projects"] = projects_with_links
            else:
                return f"[Unsupported file type: {filename}]", [], meta
            file_count += 1
            llm_response = llm_json_response(result["text"], structured_links)
            for key, value in llm_response.items():
                if key == "system_prompt":
                    sys_prom = value
                if key == "json_response":
                    json_resp = value

            with open(Json_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "messages": [
                        {"role": "system", "content": sys_prom},
                        {"role": "user", "content": result["text"]},
                        {"role": "assistant", "content": json_resp}
                    ]
                }, f, ensure_ascii=False, indent=4
            )
        print(f"\nMeta: {meta},\nFile_count: {file_count}, \nFiles not processed: {file_not_processed}")

    except Exception as e:
        file_not_processed.append(filename)
        return f"[File extraction failed: {e}]", [], {}, file_not_processed



if __name__ == "__main__":
    main()
    