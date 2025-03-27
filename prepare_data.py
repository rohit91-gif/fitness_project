import torch
import numpy as np

def create_sample_data():
    # Create sample data matching pose detection features
    num_samples = 1000
    input_features = 99  # 33 landmarks × 3 coordinates (x, y, z)
    num_classes = 3     # squats, deadlift, benchpress
    
    X = torch.randn(num_samples, input_features)
    y = torch.randint(0, num_classes, (num_samples,))
    
    torch.save(X, 'X_train.pt')
    torch.save(y, 'y_train.pt')
    print("Sample data created with correct dimensions!")

if __name__ == "__main__":
    create_sample_data()