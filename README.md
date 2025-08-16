# 📸 Image Captioning using Deep Learning

This project implements an **Image Captioning System** that automatically generates descriptive captions for images.  
It combines a **Convolutional Neural Network (CNN – InceptionV3)** for feature extraction with a **Recurrent Neural Network (RNN – LSTM)** for natural language generation.  
The model is trained and evaluated on the **Flickr8k dataset**, with performance measured using **BLEU scores**.

---

## 📁 Dataset
- **Source:** [Flickr8k Dataset](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)  
- **Total Images:** 8,000  
- **Captions:** 5 human-annotated captions per image  
- **Special Tokens:**  
  - `startseq` → Start of caption  
  - `endseq` → End of caption  

---

## 📌 Model Workflow

| Step | Description |
|------|-------------|
| **1. Feature Extraction** | Images passed through **InceptionV3** → 2048-dim feature vector |
| **2. Caption Preprocessing** | Cleaned, tokenized, and padded with start/end tokens |
| **3. Model Architecture** | Encoder (Dense layer) + Decoder (Embedding + LSTM) + Softmax |
| **4. Training** | Input: (Image features + partial caption) → Predict next word |
| **5. Inference** | Start with `startseq`, predict words until `endseq` |

---

## 📊 Algorithms & Techniques
- **CNN (InceptionV3)** → Feature extraction  
- **RNN (LSTM)** → Caption generation  
- **BLEU Scores (1–4)** → Evaluation metric  

---

## 📌 Key Features
- Uses **InceptionV3 (pre-trained on ImageNet)** for robust visual features  
- Generates captions via **LSTM** sequence modeling  
- Evaluation with **BLEU-1 to BLEU-4**  
- Deployable via **Flask Web App** (optional ngrok for sharing)  
- Simple **image upload interface** for inference  

---

## 🔧 Tech Stack
- **Programming Language:** Python  
- **Libraries:** `TensorFlow/Keras`, `NLTK`, `Flask`, `Bootstrap`, `Ngrok`  

---

## 📈 Model Performance Summary (BLEU Scores)

| Metric   | Score |
|----------|-------|
| **BLEU-1** | ~0.58 |
| **BLEU-2** | ~0.36 |
| **BLEU-3** | ~0.23 |
| **BLEU-4** | ~0.14 |

---

## 📊 Visualization
_Comparison of BLEU scores across n-grams_  

**Example 
<img width="940" height="581" alt="image" src="https://github.com/user-attachments/assets/40b13055-14d0-4ad0-844e-661be111cfd4" />

Output:**  
<img width="940" height="549" alt="image" src="https://github.com/user-attachments/assets/b8ced75f-0343-43ac-8ba5-47ef7f12b812" />


## 🚀 How to Run the Project

1. Install dependencies
   pip install -r requirements.txt

2. Run Jupyter Notebook (training & testing)
   jupyter notebook notebook.ipynb

3. OR run the Flask app
   python app.py

---

## 🔮 Future Improvements
- Implement Beam Search for improved fluency
- Add Attention Mechanism / Transformers
- Train on larger datasets (MS-COCO) 
- Fine-tune CNN layers for better features
- Integrate GPT-based caption refinement

---

## 📄 License
This project is licensed under the **MIT License**.

---

## 👤 Author
**Kadambala Eitish**  
📧 eitishkadambala@gmail.com 
🔗 [GitHub](https://github.com/Eitish24) | [LinkedIn](https://www.linkedin.com/in/kadambala-eitish0509/)





