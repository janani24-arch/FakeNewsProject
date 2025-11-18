# evaluate_deberta.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# --------------------------
# 1️⃣ Load the model & tokenizer
# --------------------------
model_path = "saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# --------------------------
# 2️⃣ Input text
# --------------------------
text = ["Reports show over 100 China-linked fake news websites operating in many countries"]
# --------------------------
# 3️⃣ Tokenize the text
# --------------------------
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)

# --------------------------
# 4️⃣ Run model prediction
# --------------------------
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1)  # Convert logits → probabilities
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item() * 100  # percentage confidence

# --------------------------
# 5️⃣ Show result
# --------------------------
if pred == 1:
    print(f" Predicted: REAL NEWS ({confidence:.2f}% confidence)")
else:
    print(f" Predicted: FAKE NEWS ({confidence:.2f}% confidence)")
