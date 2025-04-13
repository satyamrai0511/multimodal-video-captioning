# Contributing to Multimodal Video Captioning

Thanks for your interest in contributing! Follow the steps below to get started.

---

## 🛠️ Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/multimodal-video-captioning.git
   cd multimodal-video-captioning
   ```

2. **Create and activate a virtual environment**
   ```bash
   conda create -n mvcap python=3.10 -y
   conda activate mvcap
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📦 Folder Structure

- `main.py`: Entry point for running the full captioning pipeline
- `models/`: Model components (diffusion, fusion, decoder)
- `data/`: Input/output handling and pre-processing
- `output/`: Folder for saving generated captions
- `sample_video_1.mp4` / `sample_audio_1.mp3`: Demo files

---

## 📄 Guidelines

- Write clean, well-commented code
- Use descriptive commit messages
- Follow existing file and function naming conventions
- Test your code before submitting a pull request

---

## 🚀 Submitting a Pull Request

1. Fork the repository
2. Create a new branch (`git checkout -b feature-xyz`)
3. Make your changes
4. Push to your fork (`git push origin feature-xyz`)
5. Open a pull request to the `main` branch

---

## 🙋 Need Help?

If you have questions or need guidance, feel free to open an issue or contact the maintainer directly.

