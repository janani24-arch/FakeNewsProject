from transformers import AutoTokenizer, AutoModelForSequenceClassification, CLIPProcessor, CLIPModel
import torch
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# ======================
# CONFIG
# ======================
alpha = 0.7                 # weight for text (used for reference if needed)
beta = 0.3                  # weight for image-text (used for reference if needed)
text_threshold = 0.5        # threshold for text confidence to be considered REAL
similarity_threshold = 0.3  # minimum similarity for image-text consistency
max_length = 128            # DeBERTa tokenizer max length

# ======================
# DEVICE
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ======================
# LOAD MODELS
# ======================
print("Loading DeBERTa text model...")
text_tokenizer = AutoTokenizer.from_pretrained("saved_model")
text_model = AutoModelForSequenceClassification.from_pretrained("saved_model").to(device)
text_model.eval()

print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# ======================
# FUNCTIONS
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

def evaluate_image_caption(image_path, caption):
    image = Image.open(image_path).convert("RGB").resize((224,224))
    inputs = clip_processor(text=[caption], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clip_model(**inputs)
        similarity = torch.cosine_similarity(outputs.image_embeds, outputs.text_embeds)
    return similarity.item()

def show_results(image_path, caption, text_label, text_conf, similarity, final_label):
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Multimodal Fake News Detector (GPU, Parallel, Threshold)", fontsize=14, pad=20)
    plt.text(0, -10, f"Caption: {caption}", wrap=True, fontsize=10)
    plt.figtext(
        0.1, 0.05,
        f"Text Model: {text_label} ({text_conf*100:.2f}%)\n"
        f"Image Similarity: {similarity:.4f}\n"
        f"Final Prediction: {final_label}",
        fontsize=10, ha="left"
    )
    plt.tight_layout()
    plt.show()

# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    text_input = input("Enter the news text or caption: ")
    image_path = input("Enter the image file path (or press Enter to skip): ").strip()

    # Run models in parallel
    with ThreadPoolExecutor() as executor:
        text_future = executor.submit(evaluate_text, text_input)
        image_future = executor.submit(evaluate_image_caption, image_path, text_input) if image_path else None

        text_label, text_conf = text_future.result()
        similarity_score = image_future.result() if image_future else None

    # Print text results
    print(f"\nText Model Prediction: {text_label} ({text_conf*100:.2f}% confidence)")

    # Threshold-based final decision
    if similarity_score is not None:
        print(f"Image-Caption Similarity: {similarity_score:.4f}")

        if text_conf >= text_threshold and similarity_score >= similarity_threshold:
            final_label = "REAL NEWS"
        else:
            final_label = "FAKE NEWS"

        print(f"Final Combined Decision: {final_label}")
        show_results(image_path, text_input, text_label, text_conf, similarity_score, final_label)
    else:
        # Text-only prediction
        final_label = "REAL NEWS" if text_conf >= text_threshold else "FAKE NEWS"
        print(f"Final Decision (Text only): {final_label}")
