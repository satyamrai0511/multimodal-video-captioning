def align_video_audio_features(clip_features, fps, whisper_segments):
    """
    Aligns CLIP video frame features with Whisper transcript segments.

    Args:
        clip_features (np.ndarray): shape (num_frames, 512)
        fps (int): frame rate used to extract video frames
        whisper_segments (list): Whisper segments with 'start', 'end', 'text'

    Returns:
        List of dicts:
        {
            frame_index: int,
            timestamp: float,
            clip_features: list,
            audio_text: str
        }
    """
    aligned = []
    for i, feature in enumerate(clip_features):
        timestamp = i / fps

        # Find matching transcript segment
        active_segment = next(
            (seg for seg in whisper_segments if seg["start"] <= timestamp < seg["end"]),
            None
        )

        aligned.append({
            "frame_index": i,
            "timestamp": round(timestamp, 2),
            "clip_features": feature.tolist(),  # convert numpy to plain list
            "audio_text": active_segment["text"] if active_segment else ""
        })

    return aligned
