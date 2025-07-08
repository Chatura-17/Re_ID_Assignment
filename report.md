# Re-Identification in a Single Feed Report

# Intern: Chatura  
**Assignment**: Re-Identification in a Single Feed
**Video**: `15sec_input_720p.mp4`  
**Model**: Fine-tuned Ultralytics YOLOv11  
**Submission Mode**: GitHub Repository



# 1.  Approach and Methodology

The objective was to detect players in a short soccer clip, assign unique IDs, and maintain those IDs consistently — even when players leave and re-enter the frame.

#  Pipeline Breakdown:

1. **Object Detection**  
   - Used the provided fine-tuned `best.pt` model based on YOLOv11.
   - Detected players frame-by-frame using `model.predict()`.

2. **Player Feature Extraction**  
   - Calculated bounding box centers and RGB histograms for each detected player.

3. **Re-Identification and Tracking**  
   - Compared new detections with previously tracked players using:
     - **Euclidean Distance** — for spatial consistency.
     - **Cosine Similarity on Color Histograms** — for visual consistency.
   - Assigned the same ID if a close-enough match was found.
   - Assigned a new ID if no match passed both thresholds.

4. **Output Generation**  
   - Annotated each frame with bounding boxes and player IDs.
   - Saved:
     - 📹 `output_annotated.avi`: video with visualized tracking.
     - 📄 `player_tracking.csv`: log of player IDs per frame.



## 2.  Techniques Tried and Outcomes

| Technique                                 | Outcome                                                               |
|-------------------------------------------|------------------------------------------------------------------------|
| YOLOv11 object detection                  | Robust and accurate player detection throughout the clip               |
| Euclidean distance for tracking           | Worked well when players moved smoothly and didn't cross over          |
| RGB histogram + cosine similarity         | Helped distinguish visually similar but spatially distinct players     |
| Frame-by-frame sequential processing      | Enabled real-time-like ID tracking with minimal complexity             |



# 3.  Challenges Encountered

- Players close to each other with similar colors sometimes caused temporary ID confusion.
- Bounding boxes occasionally missed partially occluded players.
- Running on CPU in Google Colab added some latency.



# 4.  Incomplete Parts & Future Improvements

While the solution works well for a short clip with simple player movement, several enhancements are possible:

-  Use **Kalman Filters** or **Optical Flow** for smoother motion modeling.
-  Incorporate **Deep SORT** or a **Siamese Network** for stronger re-identification.
-  Switch to GPU-based batch inference for faster execution.
-  Apply temporal consistency checks to reduce ID switches.


