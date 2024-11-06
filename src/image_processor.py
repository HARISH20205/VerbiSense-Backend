import easyocr
import requests
import io
from PIL import Image
from typing import List, Dict, Any
import os
import numpy as np
from gradio_client import Client


def process_image_file(image_url: str) -> List[Dict[str, Any]]:
    # Fetch the image content from the URL
    response = requests.get(image_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Load the image from the response content using PIL
        image_stream = io.BytesIO(response.content)
        image = Image.open(image_stream)

        # Convert the image to a NumPy array, which is supported by EasyOCR
        image_np = np.array(image)

        # Use EasyOCR to extract text from the image
        reader = easyocr.Reader(['en'])
        result = reader.readtext(image_np)

        print("*" * 50 + image_url)
        


        # Combine the extracted text from EasyOCR
        extracted_text = "\n".join([detection[1] for detection in result])

        if len(extracted_text.split())<5 :
            # Use the BLIP model for image captioning
            client = Client("HARISH20205/blip-image-caption")
            caption_result = client.predict(image_url=image_url, api_name="/predict")
            content = "\nImage Caption:\n" + str(caption_result)
            return [{
            "file_name": os.path.basename(image_url),
            "text": content,
            }]
        # Format the content
        content = "Image Data:\n" + extracted_text 

        return [{
            "file_name": os.path.basename(image_url),
            "text": content,
        }]
    else:
        return [{
            "file_name": os.path.basename(image_url),
            "text": "Failed to retrieve image.",
        }]