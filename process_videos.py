
import os
import sys
import subprocess
import urllib.request

def install_packages():
    """Install required pip packages."""
    packages = ["mediapipe", "ultralytics", "tqdm"]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

def download_model():
    """Create models directory and download the YOLOv11 model if not already present."""
    os.makedirs("models", exist_ok=True)
    url = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11m-face.pt"
    model_path = os.path.join("models", "yolov11m-face.pt")
    if not os.path.exists(model_path):
        print(f"Downloading model from {url}...")
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")
    else:
        print("Model already exists.")

def main():
    # Install packages (if not already installed)
    install_packages()

    # Download the YOLO model
    download_model()

    # Now import the packages that were installed
    import numpy as np
    import cv2
    import mediapipe as mp
    from ultralytics import YOLO
    from tqdm import tqdm

    # Function: Dynamic frame sampling
    def dynamic_frame_sampling(video_path, min_frames=311, target_frames=423, max_frames=600):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= min_frames:
            frame_indices = np.arange(0, total_frames)
        elif total_frames <= target_frames:
            frame_indices = np.linspace(0, total_frames - 1, num=target_frames, dtype=int)
        else:
            frame_indices = np.linspace(0, total_frames - 1, num=max_frames, dtype=int)
    
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frames.append(frame)
    
        cap.release()
        return np.array(frames)

        # Paths
    dataset_path = "/dataset"
    output_path = "/output"

    os.makedirs(output_path, exist_ok=True)

    # Folder → Label mapping
    video_folders = {
        "original_sequences/youtube/c23/videos": "real",
        "manipulated_sequences/Deepfakes/c23/videos": "fake",
        "manipulated_sequences/Face2Face/c23/videos": "fake",
        "manipulated_sequences/FaceShifter/c23/videos": "fake",
        "manipulated_sequences/FaceSwap/c23/videos": "fake",
        "manipulated_sequences/NeuralTextures/c23/videos": "fake"
    }

    # Load YOLOv11 model
    yolo_model_path = "./models/yolov11m-face.pt"
    yolo_model = YOLO(yolo_model_path)
    yolo_model.overrides['verbose'] = False

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=10, refine_landmarks=True
    )

    # Main processing loop
    for folder_index, (folder, class_label) in enumerate(tqdm(video_folders.items(), desc="📂 Processing folders", dynamic_ncols=False, leave=True)):
        folder_path = os.path.join(dataset_path, folder)
        save_folder = os.path.join(output_path, class_label)
        os.makedirs(save_folder, exist_ok=True)

        if os.path.exists(folder_path):
            video_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi'))])
            batch_data = []
            batch_index = 0

            video_bar = tqdm(video_files, desc=f"🎥 {folder}", position=folder_index, leave=True, mininterval=0.5)

            for i, video_file in enumerate(video_bar):
                # Skip first 101 videos if needed
                if i <= 100:
                    continue
                video_bar.set_description(f"🎬 Processing: {video_file}")
                video_path = os.path.join(folder_path, video_file)
                frames = dynamic_frame_sampling(video_path)

                video_data = []  # List to store per-frame data
                for frame in frames:
                    # YOLOv11 prediction
                    results = yolo_model.predict(frame.copy(), device='cuda', verbose=False)
                    frame_dict = {}

                    if results and results[0].boxes:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        boxes = sorted(boxes, key=lambda b: b[0])  # Sort by x1 (left to right)

                        for subj_index, box in enumerate(boxes):
                            x1, y1, x2, y2 = box.astype(int)
                            cropped = frame[y1:y2, x1:x2]
                            if cropped.size == 0:
                                continue
                            resized = cv2.resize(cropped, (256, 256))

                            face_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                            result = mp_face_mesh.process(face_rgb)

                            if result.multi_face_landmarks:
                                landmarks = []
                                for lm in result.multi_face_landmarks[0].landmark:
                                    landmarks.append([lm.x, lm.y, lm.z])
                                landmarks = np.array(landmarks).flatten()
                            else:
                                landmarks = np.zeros(468 * 3)

                            frame_dict[f"subject_{subj_index}"] = {
                                "face": resized.astype(np.uint8),
                                "landmarks": landmarks.astype(np.float16),
                                "bbox": box.astype(np.float32)
                            }

                    video_data.append(frame_dict)

                batch_data.append({
                    "video_name": os.path.splitext(video_file)[0],
                    "frames": video_data
                })

                # Save every 50 videos in one file
                if (i + 1) % 50 == 0 or (i + 1) == len(video_files):
                    save_path = os.path.join(save_folder, f"batch_{batch_index}.npy")
                    np.save(save_path, batch_data)
                    batch_data = []
                    batch_index += 1

    print("✅ All videos processed and saved in batches with subject-wise sorted YOLO+landmark data using GPU! 🚀")

if __name__ == "__main__":
    main()
