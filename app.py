import os
import numpy as np
import pickle
import base64
from flask import Flask, request, render_template_string, make_response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from extract_features import extract_feature_single_image

# Load model and tokenizer
print("Loading model and tokenizer...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
MAX_LENGTH = 34

model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
print("Loaded successfully!")

# Generate caption
def generate_caption(photo, tokenizer, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = None
        for w, idx in tokenizer.word_index.items():
            if idx == yhat:
                word = w
                break
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text.replace('startseq', '').replace('endseq', '').strip()

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    html = """
    <!doctype html>
    <html>
    <head>
        <title>Image Caption Generator</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #e6f2ff; font-family: 'Segoe UI', sans-serif; }
            .wrapper { display: flex; justify-content: center; align-items: center; height: 100vh; }
            .box { background-color: #fff; padding: 40px; border-radius: 12px;
                   box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); text-align: center; width: 100%; max-width: 500px; }
            img { max-width: 100%; height: auto; margin-top: 20px; border-radius: 10px;
                  box-shadow: 0 2px 8px rgba(0,0,0,0.1); display: none; }
        </style>
    </head>
    <body>
        <div class="wrapper">
            <div class="box">
                <h3 class="mb-4">Upload an Image to Generate a Caption</h3>
                <form method="POST" action="/predict" enctype="multipart/form-data" id="uploadForm">
                    <input class="form-control mb-3" type="file" name="image" id="imageInput" accept="image/*" required>
                    <button type="submit" class="btn btn-primary w-100">Generate Caption</button>
                </form>
                <img id="imgPreview" src="#" alt="Image Preview"/>
            </div>
        </div>
        <script>
            const imageInput = document.getElementById('imageInput');
            const imgPreview = document.getElementById('imgPreview');
            imageInput.onchange = evt => {
                const [file] = imageInput.files;
                if (file) {
                    imgPreview.src = URL.createObjectURL(file);
                    imgPreview.style.display = 'block';
                } else {
                    imgPreview.style.display = 'none';
                }
            };
        </script>
    </body>
    </html>
    """
    return make_response(render_template_string(html))

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file or file.filename == '':
        return "No image uploaded", 400

    filepath = "temp.jpg"
    file.save(filepath)

    photo = extract_feature_single_image(filepath)
    caption = generate_caption(photo, tokenizer, MAX_LENGTH)

    with open(filepath, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")
    os.remove(filepath)
    img_src = f"data:image/jpeg;base64,{img_data}"

    html = f"""
    <!doctype html>
    <html>
    <head>
        <title>Generated Caption</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ background: #d9f0ff; text-align: center; padding-top: 50px; }}
            img {{ max-width: 350px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <h4>Generated Caption:</h4>
        <p><b>{caption}</b></p>
        <img src="{img_src}" alt="Image"/>
        <br><br>
        <a href="/" class="btn btn-secondary">Try Another</a>
    </body>
    </html>
    """
    return make_response(render_template_string(html))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
