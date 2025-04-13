from src.data_loader import load_video_frames, load_audio_waveform

video_path = "sample_data/sample_video_1.mp4"
audio_path = "sample_data/sample_audio_1.mp3"

# Test video loading
frames = load_video_frames(video_path, frame_rate=1)
print(f"✅ Loaded {len(frames)} video frames.")

# Test audio loading
waveform, sr = load_audio_waveform(audio_path)
print(f"✅ Loaded audio with shape: {waveform.shape} and sample rate: {sr}")