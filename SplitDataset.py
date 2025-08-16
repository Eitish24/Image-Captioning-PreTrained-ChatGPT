# Load image IDs (keys of features)
image_ids = list(captions.keys())
random.shuffle(image_ids)

# 80% training, 20% testing
split = int(0.8 * len(image_ids))
train_ids = image_ids[:split]
test_ids = image_ids[split:]

print("Training images:", len(train_ids))
print("Testing images:", len(test_ids))
