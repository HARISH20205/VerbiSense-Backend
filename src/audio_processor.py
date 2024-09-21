import whisper
import requests
import ffmpeg
import numpy as np
from typing import List, Dict, Any

def process_audio_from_url(audio_url: str) -> List[Dict[str, Any]]:
    # Download the audio file content
    response = requests.get(audio_url, stream=True)
    response.raise_for_status()

    # Use ffmpeg to decode the audio stream
    try:
        out, _ = (
            ffmpeg
            .input('pipe:0')
            .output('pipe:1', format='f32le', acodec='pcm_f32le', ac=1, ar='16k')
            .run(input=response.raw.read(), capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    # Convert the audio to the format Whisper expects
    audio = np.frombuffer(out, np.float32).flatten()

    # Load the Whisper model
    model = whisper.load_model("base")

    # Transcribe the audio
    result = model.transcribe(audio)

    segments = []
    for segment in result["segments"]:
        segments.append({
            "file_name": audio_url.split("/")[-1],  # Extract filename from URL
            "text": segment["text"]
        })
    print("*" * 100 + "\n" + "audio Successfull")
    return segments

def process_audio_data(audio: np.ndarray, file_name: str) -> List[Dict[str, Any]]:
    # Load the Whisper model
    model = whisper.load_model("base")

    # Transcribe the audio
    result = model.transcribe(audio)

    segments = []
    for segment in result["segments"]:
        segments.append({
            "file_name": file_name,  # Ensure file_name is added
            "text": segment["text"]
        })
    print("segments")
    return segments