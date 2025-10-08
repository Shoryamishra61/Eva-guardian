# 🚀 EVA Guardian: AI-Powered Safety Assistant for Space Stations

### Live Demo - https://codeclash-20-7htdzqnw3juzr8xw3m9aqc.streamlit.app/

**🔍 A Submission for the Duality AI - Space Station Hackathon**

---

## 1. 📝 Project Description

**EVA Guardian** is an AI-powered safety monitoring system designed to enhance operational safety and situational awareness aboard space stations.

At its core is a **YOLOv8 object detection model**, trained on a fully synthetic dataset from Duality AI’s Falcon platform. The model detects three mission-critical assets:

* 🔥 Fire Extinguishers
* 🧰 Toolboxes
* 🧪 Oxygen Tanks

This end-to-end solution includes a training pipeline, feedback loop, and a **Streamlit dashboard** for live image/webcam detection and safety reports.

📈 Our final model achieved an impressive **mAP\@0.5 of 0.936**, proving the power of synthetic data for real-world applications.

---

## 2. 🛠️ Technology Stack

* **Model:** YOLOv8 (`ultralytics`)
* **Frameworks:** PyTorch, OpenCV, NumPy, PIL
* **Frontend:** Streamlit, Plotly
* **Environment:** Python 3.10 (Conda)
* **Data Platform:** Duality AI Falcon (Synthetic)

---

## 3. ✨ Key Features

* ✅ **High-Performance Detection:**
  YOLOv8 trained for 50 epochs with augmentations (mosaic, mixup, etc.). Final metrics show strong generalization to unseen data.

* 🖥️ **Interactive Dashboard (Streamlit):**
  Upload images or use your webcam to run real-time object detection. Automatically loads latest trained model.

* 📋 **Actionable Safety Report:**
  Color-coded alerts (`CRITICAL`, `WARNING`, `OK`) based on object detection — tailored for space safety applications.

* 🔄 **Human-in-the-Loop Feedback:**
  Users can draw boxes over missed objects. A feedback processor script converts annotations into retrainable data.

* 📊 **Training Metrics Viewer:**
  Expandable section to view confusion matrix, loss curves, and precision-recall charts — all generated from training.

---

## 4. 🏆 Final Model Performance

| Metric        | Score |
| ------------- | ----- |
| **mAP\@0.5**  | 0.936 |
| **Precision** | 0.967 |
| **Recall**    | 0.895 |

Trained for 50 epochs on synthetic data with custom YOLO parameters.

---

## 5. 🚀 How to Run the Project

### ✅ Step 1: Setup & Installation

**Prerequisites:**

* Anaconda / Miniconda
* Python 3.10
* NVIDIA GPU with CUDA (for training)

```bash
# Clone the repo
git clone https://github.com/your-username/eva-guardian.git
cd eva-guardian

# Create and activate the environment
conda create -n duality_ai python=3.10
conda activate duality_ai

# Install dependencies
pip install -r requirements.txt
```

---

### ▶️ Step 2: Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

The app auto-loads your latest YOLO model from the `runs/` directory. Supports image upload, webcam, and drawing feedback boxes.

---

### 🧠 Step 3 (Optional): Re-Train with New Feedback

If you’ve collected new feedback using the app and processed it using `process_feedback.py`, you can re-train the model:

```bash
python train.py --resume
```

You can also use the improved `train_resumable.py` to automatically save progress, resume training, and avoid data loss.

---

## 6. 🧩 Overcoming Technical Challenges

| Challenge               | Solution                                                |
| ----------------------- | ------------------------------------------------------- |
| **CUDA Out of Memory**  | Reduced batch size to 4                                 |
| **System RAM Overflow** | Limited dataloader workers to 2 and disabled cache      |
| **Git Push Fails**      | `.gitignore` excludes large `data/` and `runs/` folders |

---

## 7. 🔮 Future Enhancements

* ✅ **Automated Feedback Loop:**
  Auto-retraining using a monitoring service for `feedback.json`.

* ✅ **Precise JS-Based Annotation:**
  Integrate frontend JS tools for exact coordinate capture.

* ✅ **Advanced Synthetic Generation:**
  Use domain randomization via Duality Falcon to improve sim-to-real generalization.

---

## 8. 📁 Project Structure

```text
.
├── data/                  # Dataset (excluded from GitHub)
├── new_user_images/       # Processed feedback images
├── runs/                  # YOLO training outputs (excluded from GitHub)
├── app.py                 # Streamlit dashboard
├── train.py               # Main training script
├── train_resumable.py     # Auto-resumable training version 
├── predict.py             # Inference script
├── process_feedback.py    # Converts user boxes to YOLO format
├── yolo_params.yaml       # YOLOv8 config
├── requirements.txt       # Dependencies
├── Final_Report.pdf       # Submission report
└── README.md              # This file
```

---
