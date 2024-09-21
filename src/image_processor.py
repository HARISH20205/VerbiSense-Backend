import easyocr
from typing import List, Dict, Any
import os

def process_image_file(file_path:str)->List[Dict[str, Any]]:
    reader = easyocr.Reader(['en'])

    result = reader.readtext(file_path)

    content = "\n".join(map(lambda detection: detection[1], result))
    print("*"*100+"\n"+content)
    return [{
        "file_name": os.path.basename(file_path),
        "text": content,
    }]