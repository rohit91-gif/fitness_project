import numpy as np

# Create sample labels for training
def create_labels():
    # Create 1000 samples with 3 classes (0: squats, 1: deadlift, 2: benchpress)
    num_samples = 1000
    labels = np.random.randint(0, 3, size=num_samples)
    
    # Save labels
    np.save('labels.npy', labels)
    print("Labels created successfully!")

if __name__ == "__main__":
    create_labels()