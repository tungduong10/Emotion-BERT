# 🎭 MELD Emotion Intelligence: Automatic Emotion Recognition

## 🎯 Project Objective
The primary goal of this project is to build a robust Natural Language Processing (NLP) system capable of automatically detecting and classifying human emotions from conversational text. 

Understanding emotions in dialogues is a crucial step toward enhancing Human-Computer Interaction (HCI), building empathetic chatbots, and analyzing customer sentiment. This project explores the trade-offs between model accuracy and computational efficiency by comparing a heavy-weight architecture with a lightweight, distilled model.

## 📊 The Dataset: MELD
We utilized the **MELD (Multimodal EmotionLines Dataset)** for training and evaluation. 
* **Source:** MELD is an extension of the original EmotionLines dataset, featuring multi-party conversations extracted from the popular TV sitcom *Friends*.
* **Official Link:** https://www.kaggle.com/datasets/zaber666/meld-dataset
* **Classes:** The utterances are categorized into 7 distinct emotion labels: `Anger`, `Disgust`, `Fear`, `Joy`, `Neutral`, `Sadness`, and `Surprise`.

## 🧠 Methodology & Models
To tackle this classification task, we fine-tuned and evaluated two distinct Transformer-based architectures on the MELD dataset:

1. **BERT Base (Baseline):** A powerful, heavy-weight model (12 layers) used as our primary baseline to establish the upper bound of accuracy and confidence scores.
2. **DistilRoBERTa:** A lighter, faster, distilled version of RoBERTa (6 layers). We fine-tuned this to analyze the speed-accuracy trade-off, making it ideal for real-time edge deployment.

By directly comparing these two models, we aim to provide practical insights into which architecture is best suited for different real-world production environments.

## ⚠️ Important Note for Evaluators (Giảng viên lưu ý)

Due to GitHub's strict file size limit (100MB) and Git LFS bandwidth restrictions, the pre-trained Transformer models (`.safetensors`, `.bin`) and heavy dataset features are not included in this repository. **The models are securely hosted on Google Drive.**

To run the Streamlit app successfully on your local machine, please follow these exact steps:

## 📥 Installation & Setup

**Step 1: Clone the repository**
```bash
git clone [https://github.com/tungduong10/Emotion-BERT.git](https://github.com/tungduong10/Emotion-BERT.git)
cd Emotion-BERT
```
**Step 2: Install dependencies**
```
Bash
pip install -r requirements.txt
```
**Step 3: Download the Pre-trained Models**
```
Download the models folder from this Google Drive Link:
https://drive.google.com/drive/folders/1CIBYSo_uacQw2962n5A7Q06Oy0uotbJI?usp=sharing

Extract (if zipped) and place the models/ folder directly in the root directory of this project.
```
```
Ensure the structure looks exactly like this:
├── assets/
├── models/
│   ├── bert-emotion/
│   └── roberta-emotion/
├── app.py
└── requirements.txt
```
**Step 4: Run the Application**
```
streamlit run app.py
```