from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

def extract_features(directory):
    model = InceptionV3(weights='imagenet')
    model = Model(inputs=model.input, outputs=model.layers[-2].output)
    features = dict()
    for img_name in tqdm(os.listdir(directory)):
        if not img_name.lower().endswith('.jpg'):
            continue
        filename = os.path.join(directory, img_name)
        image = Image.open(filename).resize((299, 299)).convert('RGB')
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = img_name.split('.')[0]
        features[image_id] = feature.flatten()
    return features

# Assuming images_dir is defined already:
features = extract_features(images_dir)
