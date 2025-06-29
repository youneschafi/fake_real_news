# 📰 Fake vs Real News Detector

A **Streamlit web app** that classifies news articles as **fake** or **real** using a fine-tuned DistilBERT transformer model.

---

## 🚀 Live Demo

👉 [Try it in your browser](https://youneschafi-fake-real-news-app-hrudra.streamlit.app/)

---

## 📁 Project Structure

```
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── note.ipynb          # Notebook with EDA + model exploration
```

---

## 🧠 How It Works

- Uses Hugging Face `transformers` for classification
- Splits long text into 512-token chunks
- Aggregates chunk results for a final verdict
- Also supports CSV uploads for bulk source credibility analysis

---

## ⬇️ Model Weights (Not Included)

The `transformer_model/` folder is not included due to size limits.

You can download it from:

### 🔗 Hugging Face Hub:

#### Option 1 — Programmatic Upload (if you’re the owner):
```python
from huggingface_hub import HfApi
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/path/to/local/model",
    repo_id="Chafiyounes/fakenews",
    repo_type="model",
)
```

#### Option 2 — Download as ZIP:
Go to: [https://huggingface.co/Chafiyounes/fake-news](https://huggingface.co/Chafiyounes/fake-news) and download manually.

---

## 🛠 Setup Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

---

## 📦 Dependencies

Main libraries used:
- `streamlit`
- `transformers`
- `torch`
- `pandas`
- `scikit-learn`

---

## ✍️ Author

Made by [Younes Chafi](https://github.com/Youneschafi)  
Model: [Chafiyounes/fakenews on Hugging Face](https://huggingface.co/Chafiyounes/fakenews)
