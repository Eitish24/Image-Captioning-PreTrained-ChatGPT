from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        # Convert current text to sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        
        # Map integer to word
        word = None
        for w, index in tokenizer.word_index.items():
            if index == yhat:
                word = w
                break
                
        if word is None:
            break
            
        in_text += ' ' + word
        
        if word == 'endseq':
            break
            
    final_caption = in_text.split()
    final_caption = final_caption[1:-1]  # Remove startseq and endseq
    final_caption = ' '.join(final_caption)
    return final_caption
