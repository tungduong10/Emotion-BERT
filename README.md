# ⚠️ Important Note for Evaluators (Giảng viên lưu ý)

Due to GitHub's strict file size limit (100MB) and Git LFS bandwidth restrictions, the pre-trained Transformer models (`.safetensors`, `.bin`) and heavy dataset features are not included in this repository. **They are securely hosted on Google Drive.**

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