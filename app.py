import streamlit as st
import math
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "Chafiyounes/fakenews"  # Your uploaded HF model repo
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True
)

st.set_page_config(page_title="Fake vs Real News Classifier", page_icon="📰", layout="wide")
st.title("📰 Fake vs Real News Detector")

st.write(
    """
    📝 **Single Article Classification**  
    Paste a news article (or headline + body) below and click “Classify.”  
    The model splits text into 512-token chunks and gives an overall verdict.
    """
)

# === Single Text Classification ===
article_text = st.text_area(
    "Paste your article here:",
    height=300,
    placeholder="Enter full article text (no length limit)..."
)

if st.button("Classify"):
    if not article_text.strip():
        st.warning("Please paste some text above before clicking Classify.")
    else:
        with st.spinner("Classifying..."):
            tokens = tokenizer(
                article_text,
                return_overflowing_tokens=True,
                truncation=True,
                max_length=512,
                stride=50,
                return_tensors="pt"
            )

            input_ids_chunks = tokens['input_ids']
            results = []

            for input_ids in input_ids_chunks:
                chunk_text = tokenizer.decode(input_ids, skip_special_tokens=True)
                output = classifier(chunk_text)[0]
                results.append(output)

            real_score = sum(r[1]["score"] for r in results) / len(results)
            fake_score = sum(r[0]["score"] for r in results) / len(results)

            if real_score > fake_score:
                st.success(f"The article is predicted to be REAL  (confidence: {real_score:.2%})")
            else:
                st.error(f"The article is predicted to be FAKE  (confidence: {fake_score:.2%})")

        st.write("---")
        st.write("**Chunk-wise predictions:**")
        for idx, r in enumerate(results):
            st.write(f"Chunk {idx+1}: FAKE: {r[0]['score']:.2%}, REAL: {r[1]['score']:.2%}")


st.write("---")
st.write("📂 **Batch Source Credibility Check**")

csv_file = st.file_uploader("Upload a CSV file with a 'text' or 'content' column", type=["csv"])

if csv_file is not None:
    try:
        df = pd.read_csv(csv_file)
        column_name = "text" if "text" in df.columns else "content" if "content" in df.columns else None

        if not column_name:
            st.error("CSV must contain a 'text' or 'content' column.")
        else:
            with st.spinner("Processing articles..."):
                texts = df[column_name].astype(str).tolist()
                fake_count = 0

                for text in texts:
                    output = classifier(text[:512])[0]  # Take first chunk only for speed
                    pred_label = max(output, key=lambda x: x['score'])['label']
                    if pred_label.lower() == "fake":
                        fake_count += 1

                fake_ratio = fake_count / len(texts)
                credibility = "❌ Non-credible Source" if fake_ratio > 0.4 else "✅ Credible Source"

                st.write("### 🔎 Batch Analysis Result:")
                st.markdown(f"""
                - Total Articles: **{len(texts)}**  
                - Fake Articles: **{fake_count}**  
                - Fake Percentage: **{fake_ratio * 100:.2f}%**  
                - **Conclusion:** {credibility}
                """)
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
