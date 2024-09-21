import os
from typing import List, Dict, Any
import fitz  # PyMuPDF
import docx
import requests


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

def process_pdf(pdf_url: str) -> List[Dict[str, Any]]:
        
    # Fetch the PDF file content from the URL
    response = requests.get(pdf_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Load the PDF file from the response content
        pdf_stream = io.BytesIO(response.content)
        
        # Open the PDF file with PyMuPDF
        pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
        
        # Extract text from all pages
        pdf_text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)  # Load the page
            pdf_text += page.get_text("text")  # Extract text from the page
        
        # Print the extracted text
        print(pdf_text)

    else:
        print(f"Failed to fetch the PDF file. Status code: {response.status_code}")

def process_docx(file_path: str) -> List[Dict[str, Any]]:
    doc = docx.Document(file_path)
    content = "\n".join([para.text for para in doc.paragraphs])
    
    return [{
        "file_name": os.path.basename(file_path),
        "text": content,
        "page_number": 1  # DOCX no pages concept , so  1
    }]