import ffmpeg
import numpy as np
from src.audio_processor import process_audio_data
import os
def process_video_file(file_path: str):
    # Use ffmpeg to extract audio from the video file
    try:
        out, _ = (
            ffmpeg
            .input(file_path)
            .output('pipe:1', format='f32le', acodec='pcm_f32le', ac=1, ar='16k')
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to extract audio from video: {e.stderr.decode()}") from e

    # Convert the audio to the format Whisper expects
    audio = np.frombuffer(out, np.float32).flatten()

    # Pass file name to audio processor
    file_name = os.path.basename(file_path)
    result = process_audio_data(audio, file_name)    
    return result
