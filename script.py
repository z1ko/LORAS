import os
import pickle

base_path = 'data/Assembly101/processed/validation'
for item in os.listdir(base_path):
    item_path = os.path.join(base_path, item)

    with open(item_path, 'rb') as f:
        sample = pickle.load(f)
        frames = sample['fine-labels'].shape[0]

    new_item = f'{frames:06d}-{item}'
    new_item_path = os.path.join(base_path, new_item)
    print(new_item_path)

    os.rename(item_path, new_item_path)