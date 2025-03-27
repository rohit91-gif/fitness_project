import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=30):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = 0
    success = True

    while success:
        success, frame = cap.read()
        if not success:
            break

        # Save a frame every `frame_rate` frames
        if frame_count % frame_rate == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count // frame_rate} frames to {output_folder}")

# Example usage
if __name__ == "__main__":
    video_path = 'videos/your_video.mp4'  # Update with your video file path
    output_folder = 'extracted_frames'
    extract_frames(video_path, output_folder) 