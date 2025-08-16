# Define paths
dataset_path = "data/Flickr8k_Dataset"
caption_file = "data/Flickr8k.token.txt"

# Load all captions into a dictionary
def load_captions(caption_file):
    captions_dict = {}
    with open(caption_file, 'r') as file:
        for line in file:
            tokens = line.strip().split('\t')
            if len(tokens) != 2:
                continue
            img_id, caption = tokens
            img_id = img_id.split('#')[0]
            if img_id not in captions_dict:
                captions_dict[img_id] = []
            captions_dict[img_id].append('startseq ' + caption + ' endseq')
    return captions_dict

captions = load_captions(caption_file)
print("Loaded captions for", len(captions), "images.")
