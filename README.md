
Player Re-Identification in a Single Feed

🎯 Objective
Identify players in a 15-second game video using object detection, assign consistent player IDs even if they go out of frame and return, and simulate real-time re-identification.

🧠 Approach Summary

- Used a fine-tuned YOLOv11 model (`best.pt`) to detect players in each frame.
- Assigned player IDs based on:
  - Spatial closeness (Euclidean distance)
  - Color appearance similarity (RGB histogram + cosine similarity)
- If no match is found, a new ID is assigned.
- Outputs an annotated video and a CSV log of player tracks.

⚙️ Environment & Dependencies

✅ Colab Setup
No local installation needed. Just run this project in Google Colab with GPU enabled.

📦 Dependencies
Install these in your Colab notebook before running:
```bash
!pip install ultralytics opencv-python-headless numpy pandas