import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

# Load the keypoints and labels
keypoints = np.load('keypoints.npy')
labels = np.load('labels.npy')

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(keypoints, dtype=torch.float32)
y_train_tensor = torch.tensor(labels_encoded, dtype=torch.long)

# Save the data as PyTorch tensors
torch.save(X_train_tensor, 'X_train.pt')
torch.save(y_train_tensor, 'y_train.pt') 