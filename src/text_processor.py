import os
from typing import List, Dict, Any
import fitz  # PyMuPDF
import docx
import requests
import io

def process_text_file(file_url: str) -> List[Dict[str, Any]]:
    _, extension = os.path.splitext(file_url)
    extension = extension.lower()

    if extension == '.txt':
        return process_txt(file_url)
    elif extension == '.pdf':
        return process_pdf(file_url)
    elif extension == '.docx':
        return process_docx(file_url)
    else:
        raise ValueError(f"Unsupported text file type: {extension}")

def process_txt(txt_url: str) -> List[Dict[str, Any]]:
    # Fetch the TXT file content from the URL
    response = requests.get(txt_url)

    # Check if the request was successful
    if response.status_code == 200:
        content = response.text
        print("*" * 100 + "\n" + "txt Successfull")
        return [{
            "file_name": os.path.basename(txt_url),
            "text": content,
            "page_number": 1
        }]
    else:
        print(f"Failed to fetch the TXT file. Status code: {response.status_code}")

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
        print("*" * 100 + "\n" + "pdf Successfull")

        return [{
            "file_name": os.path.basename(pdf_url),
            "text": pdf_text 
        }]
    else:
        print(f"Failed to fetch the PDF file. Status code: {response.status_code}")

def process_docx(docx_url: str) -> List[Dict[str, Any]]:
    # Fetch the DOCX file content from the URL
    response = requests.get(docx_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Load the DOCX file from the response content
        docx_stream = io.BytesIO(response.content)
        
        # Open the DOCX file with python-docx
        doc = docx.Document(docx_stream)
        
        # Extract text from the DOCX file
        content = "\n".join([para.text for para in doc.paragraphs])
        print("*" * 100 + "\n" + "docx Successfull")
        return [{
            "file_name": os.path.basename(docx_url),
            "text": content,
            "page_number": 1  # DOCX doesn't have pages, so just 1
        }]
    else:
        print(f"Failed to fetch the DOCX file. Status code: {response.status_code}")
