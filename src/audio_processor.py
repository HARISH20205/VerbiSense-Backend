import os
from typing import List, Dict, Any
import whisper

def process_audio_file(file_path: str) -> List[Dict[str, Any]]:
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    
    segments = []
    for segment in result["segments"]:
        segments.append({
            "file_name": os.path.basename(file_path),
            "text": segment["text"],
            "start_time": segment["start"],
            "end_time": segment["end"]
        })
    
    return segments