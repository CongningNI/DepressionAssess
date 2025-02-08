import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define model path
MODEL_PATH = "/home/congning/APHA-H/DepressionAssess/model/bert_final_trained_model"

# Load model and tokenizer once (for efficiency)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()  # Set model to evaluation mode
    return tokenizer, model

# Perform inference (returns a regression score)
def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted score directly (regression output)
    score = outputs.logits.item()  # Extract single value from tensor

    return max(0.0, min(1.0, score))  # Ensure output is within 0-1 range

if __name__ == "__main__":
    # Load model & tokenizer once
    tokenizer, model = load_model()
    
    while True:
        text = input("Enter text for prediction (or type 'exit' to quit): ")
        if text.lower() == "exit":
            break
        score = predict(text, tokenizer, model)
        print(f"Predicted Score: {score:.4f}")
