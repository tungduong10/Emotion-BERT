import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd
import os
from PIL import Image

st.set_page_config(
    page_title="MELD Emotion Intelligence", page_icon="🎭", layout="wide"
)


# --- CACHED MODEL LOADING ---
@st.cache_resource
def load_model(model_path):
    """Load the pre-trained tokenizer and model into memory."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model


with st.sidebar:
    st.header("Settings")
    st.info("Configure the Transformer")

    model_option = st.selectbox(
        "Select Clustering Model:",
        ("BERT Base", "DistilRoBERTa"),
    )

    # Path mapping
    model_dict = {
        "BERT Base": "./models/bert-emotion",
        "DistilRoBERTa": "./models/roberta-emotion",
    }

    st.divider()
    st.markdown("### Project Metadata")
    st.write("**Dataset:** MELD v1.0")

# Define labels
target_names = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]

# --- MAIN INTERFACE ---
st.title("MELD Abstract Analysis")
st.markdown("Automatic emotion recognition system using Transformer embeddings.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Utterance")
    user_input = st.text_area(
        "Enter English dialogue text here:",
        height=250,
        placeholder="Example: Oh my God, I can't believe it!...",
    )

    if st.button("Start Analysis", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter some text!")
        else:
            with st.spinner("Processing through Transformer layers..."):
                # 1. Load resources
                path = model_dict[model_option]
                tokenizer, model = load_model(path)

                # 2. Inference logic
                inputs = tokenizer(
                    user_input,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128,
                )
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[
                        0
                    ]
                    pred_id = np.argmax(probs)

                # 3. Success & Metrics
                st.success("Analysis Complete")
                st.metric("Predicted Emotion", target_names[pred_id])
                st.caption(
                    f"Engine: {model_option} | Confidence: {probs[pred_id]*100:.2f}%"
                )

                # Lưu vào session_state để hiển thị biểu đồ bên col2 nếu cần
                st.session_state["probs"] = probs

with col2:
    st.subheader("Emotion Distribution Visualization")
    if "probs" in st.session_state:
        chart_data = pd.DataFrame(
            {
                "Emotion Label": target_names,
                "Probability (%)": [float(p * 100) for p in st.session_state["probs"]],
            }
        ).sort_values(by="Probability (%)", ascending=False)

        st.write("**Inference Confidence Levels:**")
        st.bar_chart(data=chart_data, x="Emotion Label", y="Probability (%)")

        st.dataframe(chart_data, use_container_width=True, hide_index=True)
    else:
        st.info("Awaiting input to generate emotional knowledge space visualization.")

st.divider()

st.header("📈 Model Performance Evaluation")
st.markdown("Direct comparison of Confusion Matrices between the two architectures.")

res_col1, res_col2 = st.columns(2)

with res_col1:
    st.markdown("#### BERT Base - Confusion Matrix")
    path_cm_bert = "./assets/confusion_matrix_bert-base-uncased.png"
    if os.path.exists(path_cm_bert):
        st.image(path_cm_bert, use_container_width=True)
    else:
        st.caption(f"Asset '{path_cm_bert}' not found.")

with res_col2:
    st.markdown("#### DistilRoBERTa - Confusion Matrix")
    path_cm_roberta = "./assets/confusion_matrix_distilroberta-base.png"
    if os.path.exists(path_cm_roberta):
        st.image(path_cm_roberta, use_container_width=True)
    else:
        st.caption(f"Asset '{path_cm_roberta}' not found.")

st.divider()
st.caption(
    "Natural Language Processing - University of Science and Technology of Hanoi (USTH)"
)
