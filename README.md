🖼️ Image Captioning with Pre-trained Models

This project builds a caption generation system that automatically describes images in natural language. It uses a CNN (InceptionV3) for image feature extraction and an LSTM-based RNN for sequence generation. The system is trained and evaluated on the Flickr8k dataset.

📁 Dataset

Source: Flickr8k Dataset (8,000 images + 5 captions per image)

Captions: Human-annotated textual descriptions

Usage: Training (80%) and Testing (20%)

📌 Example:
Image → a dog running in the grass
Caption → “a dog is running in the grass”

📊 Feature Descriptions (Model Input/Output)
Feature	Description	Role in Captioning
Image	Raw image (JPEG/PNG)	Input for CNN (InceptionV3)
Feature Vector	2048-dim extracted from InceptionV3	Encoded visual representation
Caption Tokens	Text captions with <start> and <end> markers	Input/output for LSTM sequence model
Vocabulary	Unique words mapped to integers	Required for embedding + generation
📊 Algorithms / Models Used

InceptionV3 (CNN) → Pre-trained on ImageNet, used for feature extraction

LSTM (RNN) → Generates captions word by word

Tokenizer → Converts captions into integer sequences

BLEU Score Evaluation → Compares generated vs. reference captions

📌 Key Features

Extracts visual features from images using transfer learning

Generates captions with LSTM-based sequence model

Evaluated using BLEU-1 to BLEU-4 scores

Provides Flask web interface for real-time captioning

Ngrok integration for sharing app publicly

🔧 Tech Stack

Programming Language: Python

Libraries: TensorFlow/Keras, NumPy, Pandas, NLTK, Matplotlib, Seaborn

Deployment: Flask, Bootstrap, Ngrok

📈 Model Performance Summary
Metric	Score
BLEU-1	0.58 (good keyword overlap)
BLEU-2	0.36
BLEU-3	0.23
BLEU-4	0.14 (sentence-level fluency)

👉 Model performs well for short captions but struggles with longer sentence structure.

📊 Visualization

Training loss curves

Caption predictions vs. ground truth

Performance comparison with BLEU metrics

🚀 How to Run the Project

Clone the repository:
