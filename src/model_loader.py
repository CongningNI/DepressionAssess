import os
import gdown
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define paths
MODEL_PATH = "model/bert_final_trained_model"
GDRIVE_FOLDER_ID = "1wP8VZxZqSZX6QVRzeAk3NnyWW-2ODJVC"  # Replace with your actual Google Drive Folder ID

def download_model_folder():
    """Force downloads the entire model folder from Google Drive."""
    print("🔍 Checking if model folder exists...")
    
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model folder already exists: {MODEL_PATH}")
        return  # Skip download

    print("🔽 Model folder not found. Downloading from Google Drive...")
    os.makedirs(MODEL_PATH, exist_ok=True)

    # Use gdown API instead of os.system
    try:
        gdown.download_folder(f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}", output=MODEL_PATH, quiet=False)
        print("✅ Model folder downloaded successfully!")
    except Exception as e:
        print(f"❌ Failed to download model folder: {e}")

def load_model():
    """Loads the fine-tuned BERT model and tokenizer."""
    download_model_folder()  # Ensure the model folder is available
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

if __name__ == "__main__":
    print("🚀 Running model_loader.py...")
    download_model_folder()
