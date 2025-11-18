from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification, CLIPProcessor, CLIPModel
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# ======================
# CONFIG
# ======================
alpha = 0.7
beta = 0.3
text_threshold = 0.5
similarity_threshold = 0.3
max_length = 128

# ======================
# DEVICE
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# LOAD MODELS
# ======================
text_tokenizer = AutoTokenizer.from_pretrained("saved_model")
text_model = AutoModelForSequenceClassification.from_pretrained("saved_model").to(device)
text_model.eval()

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",use_safetensors=True).to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",use_fast=True)
clip_model.eval()

# ======================
# HELPER FUNCTIONS
# ======================
def evaluate_text(text):
    inputs = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, pred = torch.max(probs, dim=1)
    label = "REAL" if pred.item() == 1 else "FAKE"
    return label, confidence.item()

def evaluate_image_caption(image, caption):
    image = Image.open(image).convert("RGB").resize((224,224))
    inputs = clip_processor(text=[caption], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clip_model(**inputs)
        similarity = torch.cosine_similarity(outputs.image_embeds, outputs.text_embeds)
    return similarity.item()

# ======================
# FLASK APP
# ======================
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    text_input = request.form.get("text")
    image_file = request.files.get("image")

    if not text_input.strip():
        return jsonify({"error": "Please enter news text."})

    # Run models in parallel
    with ThreadPoolExecutor() as executor:
        text_future = executor.submit(evaluate_text, text_input)
        image_future = executor.submit(evaluate_image_caption, image_file, text_input) if image_file else None

        text_label, text_confidence = text_future.result()
        similarity_score = image_future.result() if image_future else None

    # Final decision based on thresholds
    if similarity_score is not None:
        combined_score = alpha * text_confidence + beta * similarity_score
        if text_confidence >= text_threshold and similarity_score >= similarity_threshold:
            final_label = "REAL NEWS"
        else:
            final_label = "FAKE NEWS"
    else:
        combined_score = text_confidence
        final_label = "REAL NEWS" if text_confidence >= text_threshold else "FAKE NEWS"

    return jsonify({
        "text_label": text_label,
        "text_confidence": round(text_confidence*100, 2),
        "similarity": round(similarity_score, 4) if similarity_score is not None else None,
        "final_label": final_label
    })

if __name__ == "__main__":
    app.run(debug=True)
