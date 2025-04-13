import numpy as np
from src.data_loader import load_video_frames
from src.feature_extractor import extract_video_features, extract_audio_features
from src.fusion_utils import align_video_audio_features
import whisper

# === Paths ===
video_path = "sample_data/sample_video_1.mp4"
audio_path = "sample_data/sample_audio_1.mp3"
fps = 1  # Assuming 1 frame per second during video loading

# === Load and extract ===
print("🎥 Loading video frames...")
frames = load_video_frames(video_path, frame_rate=fps)
print(f"✅ Loaded {len(frames)} frames")

print("🧠 Extracting CLIP features...")
clip_features = extract_video_features(frames)

print("🎧 Transcribing audio with Whisper...")
model = whisper.load_model("base")
result = model.transcribe(audio_path, language="en", fp16=False)
segments = result["segments"]

# === Align ===
print("🔗 Aligning video frames to audio segments...")
aligned = align_video_audio_features(clip_features, fps, segments)

# === Show result ===
for item in aligned[:5]:  # Show first 5 alignments
    print(f"[{item['timestamp']}s] {item['audio_text']}")

import json
import os

# === Save aligned output ===
output_path = "aligned_data.json"
with open(output_path, "w") as f:
    json.dump(aligned, f, indent=2)

print(f"📁 Aligned data saved to: {os.path.abspath(output_path)}")
