import streamlit as st
import math
import os
from transformers import pipeline, RobertaTokenizerFast, RobertaForSequenceClassification
import streamlit as st
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "Chafiyounes/fakenews"
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
    Paste a news article (or headline + body) below and click “Classify.”  
    The model will split into 512-token chunks, classify each, and then output
    whether it’s “fake” or “real” with a confidence score.
    """
)

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
