from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
def generate_caption_greedy(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = yhat.argmax()
        word = tokenizer.index_word.get(yhat, None)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text
# Evaluate all test images
actual, predicted = list(), list()
for img_id in tqdm(test_descriptions.keys()):
    photo = features[img_id].reshape((1, 2048))
    yhat = generate_caption_greedy(model, tokenizer, photo, max_len)
    references = [d.split() for d in test_descriptions[img_id]]
    yhat_clean = yhat.split()
    actual.append(references)
    predicted.append(yhat_clean)
