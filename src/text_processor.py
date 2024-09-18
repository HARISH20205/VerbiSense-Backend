import os
from typing import List, Dict, Any
import fitz  # PyMuPDF
import docx

def process_text_file(file_path: str) -> List[Dict[str, Any]]:
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    if extension == '.txt':
        return process_txt(file_path)
    elif extension == '.pdf':
        return process_pdf(file_path)
    elif extension == '.docx':
        return process_docx(file_path)
    else:
        raise ValueError(f"Unsupported text file type: {extension}")

def process_txt(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    return [{
        "file_name": os.path.basename(file_path),
        "text": content,
        "page_number": 1
    }]

def process_pdf(file_path: str) -> List[Dict[str, Any]]:
    results = []
    
    with fitz.open(file_path) as doc:
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            results.append({
                "file_name": os.path.basename(file_path),
                "text": text,
                "page_number": page_num
            })
    
    return results

def process_docx(file_path: str) -> List[Dict[str, Any]]:
    doc = docx.Document(file_path)
    content = "\n".join([para.text for para in doc.paragraphs])
    
    return [{
        "file_name": os.path.basename(file_path),
        "text": content,
        "page_number": 1  # DOCX no pages concept , so  1
    }]