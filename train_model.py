import torch
import torch.nn as nn
import os

class FitnessModel(nn.Module):
    def __init__(self, input_size=132, num_classes=3):  # Changed from 33*4 to 132
        super(FitnessModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create and save a basic model
model = FitnessModel()
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'fitness_model.pth')

# Save the model
torch.save(model.state_dict(), model_path)

# Verify the file exists
if os.path.exists(model_path):
    print(f"Model successfully saved to: {model_path}")
    print("Model input size:", next(model.parameters()).size())
else:
    print("Error: Model file not saved!")