import torch
import torch.nn as nn
import torch.optim as optim
from model import FitnessModel
import os

# Load training data
X_train = torch.load('X_train.pt')
y_train = torch.load('y_train.pt')

# Get actual input size from data
input_size = X_train.shape[1]  # This will be 3
num_classes = len(torch.unique(y_train))  # Number of unique classes

# Initialize model with correct dimensions
model = FitnessModel(input_size=input_size, num_classes=num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), 'fitness_model.pth')
print("Model saved successfully!")