import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments





# -------------------------------
# 1. Device check (GPU or CPU)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# 2. Load datasets
# -------------------------------
# Make sure your CSVs are in this path (adjust if needed)
data_files = {
    "train": "datasets_folder/train.csv",
    "test": "datasets_folder/test.csv"
}

dataset = load_dataset("csv", data_files=data_files)

train_dataset = dataset["train"]
test_dataset = dataset["test"]



# -------------------------------
# 3. Load DeBERTa tokenizer
# -------------------------------
model_name = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)



# -------------------------------
# 4. Tokenization function
# -------------------------------
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

# -------------------------------
# 5. Apply tokenizer to datasets
# -------------------------------
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# -------------------------------
# 6. Load DeBERTa model
# -------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    trust_remote_code=True,
    use_safetensors=True  # ✅ Forces loading the safe version
)

model.to(device)

# -------------------------------
# 7. Training arguments
# -------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)


# -------------------------------
# 8. Define Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# -------------------------------
# 9. Train model
# -------------------------------
trainer.train()

# -------------------------------
# 10. Evaluate model
# -------------------------------
results = trainer.evaluate()
print(" Evaluation Results:")
print(results)

# -------------------------------
# 11. Save the model and tokenizer
# -------------------------------
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

print(" Training complete! Model saved in './saved_model'")