import easyocr
import requests
import io
from PIL import Image
from typing import List, Dict, Any
import os
import numpy as np

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

        # Combine the extracted text
        content = "\n".join(map(lambda detection: detection[1], result))

        print("*" * 100 + "\n" + "Image successfull")
        
        return [{
            "file_name": os.path.basename(image_url),
            "text": content,
        }]
    else:
        print(f"Failed to fetch the image file. Status code: {response.status_code}")
        return []
