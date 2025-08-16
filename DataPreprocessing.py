import os
import random
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from tqdm import tqdm
import pickle

# Step 1: Extract features using InceptionV3
def extract_features(directory):
    model = InceptionV3(weights='imagenet')
    model = Model(inputs=model.input, outputs=model.layers[-2].output)

    features = {}
    for name in tqdm(os.listdir(directory)):
        if name.lower().endswith(('.jpg', '.jpeg', '.png')):
            filename = os.path.join(directory, name)
            image = load_img(filename, target_size=(299, 299))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            feature = model.predict(image, verbose=0)
            image_id = name.split('.')[0]
            features[image_id] = feature
    return features

# Optional: Load from pickle if already saved
# features = pickle.load(open("features.pkl", "rb"))

# Otherwise extract and save
features = extract_features("data/Flickr8k_Dataset")
with open("features.pkl", "wb") as f:
    pickle.dump(features, f)
