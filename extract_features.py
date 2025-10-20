from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image

# Load model once at module level
base_model = InceptionV3(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

def extract_feature_single_image(img_path):
    img = Image.open(img_path).resize((299, 299)).convert('RGB')
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x, verbose=0)
    return feature
