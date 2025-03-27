import os
import cv2
import mediapipe as mp
import numpy as np

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def extract_keypoints_from_frames(frame_folder):
    keypoints_list = []
    labels = []  # Store corresponding labels

    for filename in os.listdir(frame_folder):
        if filename.endswith('.jpg'):
            frame_path = os.path.join(frame_folder, filename)
            frame = cv2.imread(frame_path)

            # Process the frame
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    keypoints.append([landmark.x, landmark.y, landmark.z])
                keypoints_list.append(keypoints)

                # Extract label from filename (assuming format: label_frame_x.jpg)
                label = filename.split('_')[0]  # Adjust based on your naming convention
                labels.append(label)

    return np.array(keypoints_list), np.array(labels)

# Example usage
frame_folder = 'extracted_frames'  # Ensure this folder contains your extracted frames
keypoints, labels = extract_keypoints_from_frames(frame_folder)

# Save the keypoints and labels for later use
np.save('keypoints.npy', keypoints)
np.save('labels.npy', labels) 