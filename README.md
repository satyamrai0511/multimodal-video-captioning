# 🎥 Multimodal Video Captioning using Diffusion and Autoregressive Models

This project generates intelligent and context-aware captions from video and audio streams by combining **diffusion models** with **multimodal autoregressive decoders**. It aims to improve temporal coherence and linguistic fluency by leveraging both visual and auditory inputs.

---

## 📌 Project Goals

- Generate meaningful captions from real-world videos
- Fuse audio and visual features for richer context
- Use denoising diffusion models to enhance feature representations
- Decode captions using autoregressive language models

---

## 🧠 Model Overview

1. **Feature Extraction**
   - Visual features from video frames (e.g., via CNN/ViT)
   - Audio features from speech using pretrained models (e.g., Wav2Vec2)

2. **Diffusion-Based Embedding Denoising**
   - Temporal and contextual refinement

3. **Multimodal Fusion**
   - Adaptive attention across visual and audio streams

4. **Autoregressive Captioning**
   - GPT-style decoder generates fluent natural language captions

---

## 📁 Folder Structure

```
multimodal-video-captioning/
├── models/                    # Core model code
├── data/                      # Data loaders and preprocessors
├── utils/                     # Utility scripts
├── samples/                   # Placeholder for input files
│   ├── sample_video_1.mp4     # (not included due to size)
│   ├── sample_audio_1.mp3     # (not included due to size)
│   └── sample_caption.txt     # Sample output (optional)
├── output/                    # Caption results
├── main.py                    # Main script
├── test.py                    # Test script
├── requirements.txt           # Dependencies
├── LICENSE
├── .gitignore
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/multimodal-video-captioning.git
cd multimodal-video-captioning
```

### 2. Create and Activate Environment

```bash
conda create -n mvcap python=3.10 -y
conda activate mvcap
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

To run the model, place your input files in the root or `samples/` folder:

- `sample_video_1.mp4`
- `sample_audio_1.mp3`

> ⚠️ Large media files are excluded from the repository due to GitHub size limits.

Then run the pipeline:

```bash
python main.py --video samples/sample_video_1.mp4 --audio samples/sample_audio_1.mp3
```

The generated caption will be saved in:

```
output/captions.txt
```

---

## 🧪 Testing

Use the `test.py` script for testing with sample inputs:

```bash
python test.py
```

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).

---

## 🤝 Contributing

Contributions are welcome!

- Fork the repository
- Create a feature branch (`git checkout -b feature-name`)
- Make changes and commit
- Submit a pull request to `main`

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for details.

---

## 🙋 Maintainer

Maintained by Satyam Rai
For questions or feedback, contact: satyamrai@outlook.com

---
