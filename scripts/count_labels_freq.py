import os
import glob
import pickle
import numpy as np
import tqdm

def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

total_frames = 0
counter = [
    np.array([1.0] * 25), # verbs
    np.array([1.0] * 91)  # classes
]

path_to_dataset = './data/Assembly101'
for filepath in tqdm.tqdm(list(glob.iglob(path_to_dataset + '/**/*.pkl', recursive=True))):
    with open(filepath, 'rb') as f:
        labels = np.int64(pickle.load(f)['fine-labels'])

    for category_id, num_classes in enumerate([25, 91]):
        count = np.bincount(labels[:, category_id + 1], minlength=num_classes)
        counter[category_id] += count
        total_frames += labels.shape[0]

weights = [ 100000.0 / count for count in counter ]

print(counter)
print(weights)

with open('fine-labels-weights.pkl', 'wb') as f:
    pickle.dump(weights, f)