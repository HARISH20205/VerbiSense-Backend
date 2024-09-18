import subprocess
from src.audio_processor import process_audio_file
import os
def process_video_file(file_path:str):
    output_file = "files/audio1.mp3"
    i = 1
    while os.path.exists(output_file):
        i += 1
        output_file = f"files/audio{i}.mp3"

    cmd = ["ffmpeg",
           "-i",
           file_path,
           "-vn", 
           "-acodec","libmp3lame",
           "-ab","192k",
           "-ar","44100",
           "-y",
           output_file
    ]
    try:
        subprocess.run(cmd, check=True)
        result = process_audio_file("files/audio1.mp3")
        return result
    except Exception as e:
        print(f"Error processing audio: {e}")
        return []