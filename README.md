# Depression Assessment Model (BERT Regression)

This repository provides a fine-tuned BERT model that predicts the probability that a given text is **"Asking a Question" (1.0) vs. "Sharing an Experience" (0.0)**.

## Project Structure

```
DepressionAssess/
│── model/                   # Stores the downloaded model files
│── src/                     # Contains all scripts
│   ├── model_loader.py      # Downloads & loads the model
│   ├── inference.py         # Runs inference on input text
│── notebooks/               # Jupyter notebooks for testing
│── requirements.txt         # Required dependencies
│── README.md                # Documentation
```

## 🚀 How to Set Up & Use

### 1️⃣ Clone This Repository
If you haven't already:
```
git clone https://github.com/YOUR_GITHUB_USERNAME/DepressionAssess.git
cd DepressionAssess
```

### 2️⃣ Create a Virtual Environment
To keep dependencies clean:
```
python -m venv env
source env/bin/activate  # (On Linux/macOS)
# On Windows: env\Scriptsctivate
```

### 3️⃣ Install Dependencies
```
pip install -r requirements.txt
```

### 4️⃣ Download the Fine-Tuned Model from Google Drive
Since the model is stored in **Google Drive**, we use `gdown` to download it.

Run:
```
python src/model_loader.py
```
This will:
✅ **Download the model folder** (`bert_final_trained_model/`).  
✅ **Store it in the `model/` directory**.  
✅ **Ensure all necessary files (`model.safetensors`, `config.json`, `vocab.txt`) are available**.

### 5️⃣ Run Inference
Now, run:
```
python src/inference.py
```
You can enter sentences, and it will return a score:
```
Enter text for prediction (or type 'exit' to quit): I need help, I have a question.
Predicted Score: 0.7963

Enter text for prediction (or type 'exit' to quit): I have an experience to share.
Predicted Score: 0.0464
```

✅ **A score closer to 1.0** → The text is likely asking a question.  
✅ **A score closer to 0.0** → The text is more of a shared experience.

## 🎯 Notes
- If the model fails to load, ensure `model_loader.py` **has the correct Google Drive file ID**.
- The model **must be in the correct folder** (`model/bert_final_trained_model/`).

## 🤖 Future Improvements
- Deploy as a **Flask API** for real-time scoring.
- Train with **more diverse datasets** for better generalization.
- Add **visualization of prediction distributions**.

## 📩 Contact
For questions, open an **Issue** or reach out at:
📧 **Email:** congning.ni@vanderbilt.edu 
📂 **GitHub:** [CongningNI](https://github.com/CongningNI)


