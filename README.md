# Multimodal Fake News Detector

**A Python-based AI application to detect fake news using text and images, combining DeBERTa and CLIP models.**

---

## Overview

This project uses a **multimodal approach** for fake news detection:

1. **Text Classification**  
   - Uses a pretrained **DeBERTa** model to classify news text as **Fake** or **Real**.  
   - Outputs a confidence score (0–1).  

2. **Image-Text Consistency**  
   - Uses **CLIP** (OpenAI) to measure similarity between uploaded news images and their captions.  
   - Computes a similarity score (0–1).  

3. **Final Prediction**  
   - Combines text and image scores using:  

   ```
   Score_final = α × Score_text + β × Score_image-text
   ```
   - Thresholding is applied to decide **FAKE** or **REAL** news.

---

## Features

- **Multimodal input:** Accepts text and optional images.  
- **Threshold-based prediction:** Configurable α, β, and threshold.  
- **GPU compatible:** Supports fast inference on GPU.  
- **Visualization:** Shows image, caption, text confidence, similarity, and final prediction.  
- **Batch testing compatible:** Can be extended to test multiple news + images.  

---

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd multimodal-fake-news-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have your pretrained **DeBERTa model** saved in `saved_model/`.

---

## Usage

### 1. Single prediction
```bash
python multimodal_fake_news_detector_gpu.py
```
- Enter the news text when prompted.  
- Enter the image file path (optional) or press Enter to skip.  
- The script outputs:  
  - Text prediction (DeBERTa)  
  - Image-text similarity (CLIP)  
  - Final combined score and label  
  - Visualization if an image is provided  

---

### 2. Batch testing (optional)
- You can create a Python script to load multiple news texts + images from a CSV and run predictions in a loop for fast evaluation.  

---

## Configuration

- **α (alpha):** Weight for text prediction (default `0.7`)  
- **β (beta):** Weight for image-text similarity (default `0.3`)  
- **Threshold:** Final score threshold to decide FAKE/REAL (default `0.5`)  
- **max_length:** Max token length for DeBERTa tokenizer (default `128`)  

---

## Requirements

- Python 3.10+  
- `torch`  
- `transformers`  
- `datasets`  
- `Pillow`  
- `matplotlib`  

---

## Example

```text
Enter the news text or caption: China launches a new space mission to Mars.
Enter the image file path (or press Enter to skip): mars.jpg

Text Model Prediction: REAL (87.35% confidence)
Image-Caption Similarity: 0.72
Final Combined Score: 0.86
Final Combined Decision: REAL NEWS
```

---

## License

This project is for **educational purposes only**.

