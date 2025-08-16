from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from keras.preprocessing.sequence import pad_sequences

# Load extracted features
with open("features.pkl", "rb") as f:
    all_features = pickle.load(f)

# Filter only training features
train_features = {k: all_features[k] for k in train_ids}

# Prepare the tokenizer
all_captions = []
for img_id in train_ids:
    for cap in captions[img_id]:
        all_captions.append("startseq " + cap + " endseq")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(c.split()) for c in all_captions)

# Save tokenizer for later use
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Create data generator
def data_generator(captions, features, tokenizer, max_length, vocab_size):
    while True:
        for key, cap_list in captions.items():
            photo = features[key][0]
            for cap in cap_list:
                cap = 'startseq ' + cap + ' endseq'
                seq = tokenizer.texts_to_sequences([cap])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    yield [[photo, in_seq], out_seq]

# Define the model architecture
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

model = define_model(vocab_size, max_length)

# Train the model using generator
steps = sum(len(captions[k]) for k in train_ids)
generator = data_generator(captions, train_features, tokenizer, max_length, vocab_size)

model.fit(generator, epochs=20, steps_per_epoch=steps, verbose=1)

# Save the trained model
model.save("model.keras")
