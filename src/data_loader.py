import cv2
import librosa

def load_video_frames(video_path, frame_rate=1):
    """
    Extract frames from a video at a given frame rate.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps // frame_rate) if fps else 1

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frames.append(frame)
        count += 1

    cap.release()
    return frames

def load_audio_waveform(audio_path, sr=16000):
    """
    Load audio waveform using librosa.
    """
    waveform, sample_rate = librosa.load(audio_path, sr=sr)
    return waveform, sample_rate