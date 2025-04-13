import torch
import clip
import whisper
from PIL import Image
import numpy as np

# ========== DEVICE SETUP ==========
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== CLIP MODEL ==========
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def extract_video_features(frames):
    """
    Extract CLIP visual embeddings from video frames.
    Args:
        frames (list of numpy arrays): BGR video frames
    Returns:
        numpy array of shape (num_frames, 512)
    """
    images = [preprocess(Image.fromarray(frame[:, :, ::-1])).unsqueeze(0) for frame in frames]
    images = torch.cat(images).to(device)

    with torch.no_grad():
        features = clip_model.encode_image(images)

    return features.cpu().numpy()


# ========== WHISPER MODEL ==========
whisper_model = whisper.load_model("base")

def extract_audio_features(audio_path):
    """
    Transcribe audio using Whisper and return text.
    Includes debug print of the raw result.
    """
    print(f"🔍 Transcribing: {audio_path}")
    result = whisper_model.transcribe(audio_path, language='en', fp16=False)
    print("📦 Raw Whisper output:", result)  # Debug info
    return result["text"]
