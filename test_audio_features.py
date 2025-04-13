from src.feature_extractor import extract_audio_features

audio_path = "sample_data/sample_audio_1.mp3"

print("🎧 Extracting Whisper features...")
text = extract_audio_features(audio_path)
print("📝 Transcribed Audio:")
print(text)
