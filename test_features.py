from src.data_loader import load_video_frames
from src.feature_extractor import extract_video_features

video_path = "sample_data/sample_video_1.mp4"

# Load frames
frames = load_video_frames(video_path, frame_rate=1)
print(f"✅ Loaded {len(frames)} video frames.")

# Extract features
features = extract_video_features(frames)
print(f"✅ Extracted CLIP features with shape: {features.shape}")