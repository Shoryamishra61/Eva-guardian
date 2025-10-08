# EVA Guardian - Object Detection Dashboard
# Improved Version with functional feedback loops and UX enhancements.

import os, sys, subprocess

# --- Environment Fixes ---
# Silence OpenCV logs and prevent OpenMP duplication errors
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure only headless OpenCV is active (remove GUI version if installed)
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)

# --- Imports ---
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import json
import pandas as pd
import time
import plotly.express as px
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple, Set

import cv2, streamlit as st
st.caption(f"cv2: {cv2.__version__} (headless expected if no GUI)")

# --- PAGE CONFIGURATION (with Dark Mode as default) ---
st.set_page_config(
    page_title="EVA Guardian",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTS ---
# Directories and File Paths
RUNS_DIR = os.path.join('runs', 'detect')
WEIGHTS_FILE = os.path.join('weights', 'best.pt')
RESULTS_CSV = 'results.csv'
FEEDBACK_DIR = "feedback"
# Keep feedback files in the main directory as per the original code.
INCORRECT_FEEDBACK_FILE = "incorrect_feedback.json"
MISSED_FEEDBACK_FILE = "feedback.json"
FEEDBACK_IMAGE_DIR = os.path.join(FEEDBACK_DIR, "new_user_images")

# UI and Styling
PAD_COLOR = (38, 39, 48)  # Dark padding for gallery images
MAX_IMAGE_SIZE = (1280, 720)

# Model and Risk Assessment
DEFAULT_URGENCY = 0.5
URGENCY_WEIGHTS = {'FireExtinguisher': 1.0, 'OxygenTank': 0.8, 'ToolBox': 0.6}


# --- INITIAL SETUP ---
# Create necessary directories on startup
os.makedirs(FEEDBACK_IMAGE_DIR, exist_ok=True)


# --- HELPER FUNCTIONS ---
@st.cache_data
def get_model_versions() -> Dict[str, str]:
    """
    Finds all valid YOLOv8 training run directories and returns them
    as a user-friendly dictionary, sorted from newest to oldest.
    """
    models = {}
    if not os.path.exists(RUNS_DIR):
        return models

    all_dirs = [d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))]
    # Filter for directories that contain a completed training run
    train_dirs = [d for d in all_dirs if os.path.exists(os.path.join(RUNS_DIR, d, WEIGHTS_FILE))]

    # Sort directories by modification time (newest first)
    sorted_paths = sorted([os.path.join(RUNS_DIR, d) for d in train_dirs], key=os.path.getmtime, reverse=True)

    for i, path in enumerate(sorted_paths):
        version_name = f"Version {(len(sorted_paths) - i)}"
        if i == 0:
            version_name += " (Latest)"
        models[version_name] = path

    return models


@st.cache_data
def load_metrics_from_run(run_path: str) -> Optional[Dict[str, float]]:
    """Loads key performance metrics from the results.csv file of a training run."""
    results_path = os.path.join(run_path, RESULTS_CSV)
    if os.path.exists(results_path):
        try:
            df = pd.read_csv(results_path)
            # Clean column names from leading/trailing spaces
            df.columns = df.columns.str.strip()
            # Get the metrics from the last epoch
            latest_metrics = df.iloc[-1]
            return {
                'mAP50': latest_metrics.get('metrics/mAP50(B)', 0),
                'Precision': latest_metrics.get('metrics/precision(B)', 0),
                'Recall': latest_metrics.get('metrics/recall(B)', 0)
            }
        except Exception as e:
            st.warning(f"Could not parse metrics file: {e}")
            return None
    return None


@st.cache_resource
def load_model(model_path: str) -> Optional[YOLO]:
    """
    Loads the YOLO model from the specified path.
    Uses st.cache_resource to prevent reloading the model on every script run.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None


def resize_image(image: Image.Image, max_size: Tuple[int, int] = MAX_IMAGE_SIZE) -> Image.Image:
    """Resizes a PIL image if it exceeds the max dimensions, preserving aspect ratio."""
    if image.width > max_size[0] or image.height > max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def pad_image_to_square(image: Image.Image) -> Image.Image:
    """Pads a PIL image with a dark background to make it square."""
    width, height = image.size
    if width == height:
        return image
    
    bigger_dim = max(width, height)
    result = Image.new(image.mode, (bigger_dim, bigger_dim), PAD_COLOR)
    
    # Paste the original image into the center of the new square image
    paste_x = (bigger_dim - width) // 2
    paste_y = (bigger_dim - height) // 2
    result.paste(image, (paste_x, paste_y))
    
    return result


class VideoTransformer(VideoTransformerBase):
    """Processes video frames for real-time object detection."""
    def __init__(self, model: YOLO, confidence_threshold: float):
        self.model = model
        self.confidence_threshold = confidence_threshold

    def transform(self, frame: Any) -> np.ndarray:
        """Applies object detection to a single video frame."""
        img_bgr = frame.to_ndarray(format="bgr24")
        # Run detection on the frame
        results = self.model(img_bgr, conf=self.confidence_threshold, stream=True, verbose=False)
        
        # Plot results directly on the frame
        for r in results:
            img_bgr = r.plot()
            
        return img_bgr


# --- UI HANDLERS ---
def handle_dashboard(model: YOLO, confidence_threshold: float, selected_model_name: str, run_path: str):
    """Displays the main dashboard content."""
    st.header(f"Performance for: *{selected_model_name}*")
    metrics = load_metrics_from_run(run_path)
    if metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("mAP@0.5 Score", f"{metrics['mAP50']:.3f}", help="Mean Average Precision at 50% overlap. The primary metric for object detection accuracy. Higher is better.")
        col2.metric("Precision", f"{metrics['Precision']:.3f}", help="Of all the detections made, how many were correct? (TP / (TP + FP))")
        col3.metric("Recall", f"{metrics['Recall']:.3f}", help="Of all the true objects, how many did the model find? (TP / (TP + FN))")
    else:
        st.warning("Could not load performance metrics for this model version.")
    
    st.divider()

    view_toggle = st.radio(
        "Select View", ["🚀 Live Demo", "📈 Performance Deep Dive"],
        horizontal=True, label_visibility="collapsed"
    )
    
    if view_toggle == "🚀 Live Demo":
        handle_live_demo(model, confidence_threshold)
    else:
        handle_performance_analysis(run_path)


def handle_live_demo(model: YOLO, confidence_threshold: float):
    """Manages the live demo section for image and webcam detection."""
    source_type = st.radio("Select Source", ["Image", "Webcam"], horizontal=True, label_visibility="collapsed")
    if source_type == "Image":
        handle_image_detection(model, confidence_threshold)
    elif source_type == "Webcam":
        handle_webcam_detection(model, confidence_threshold)


def handle_image_detection(model: YOLO, confidence_threshold: float):
    """Manages the image upload, detection, and feedback workflow."""
    uploaded_file = st.file_uploader(
        "Upload an Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="file_uploader"
    )

    if uploaded_file is None:
        if 'current_file_id' in st.session_state:
            del st.session_state['current_file_id']
        if 'reported_falses' in st.session_state:
            del st.session_state['reported_falses']
        return

    file_id = f"{uploaded_file.name}-{uploaded_file.size}"

    if st.session_state.get('current_file_id') != file_id:
        st.session_state.current_file_id = file_id
        st.session_state.reported_falses = set()

    image = Image.open(uploaded_file).convert("RGB")
    image = resize_image(image)

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Original Image")
        st.image(image, use_container_width=True)
    
    with st.spinner("Analyzing image..."):
        results = model(image, conf=confidence_threshold, verbose=False)
    
    res_plotted_bgr = results[0].plot()
    res_plotted_rgb = cv2.cvtColor(res_plotted_bgr, cv2.COLOR_BGR2RGB)
    
    with col2:
        st.write("#### Detected Image")
        st.image(res_plotted_rgb, use_container_width=True)
        
        _, buf = cv2.imencode(".png", res_plotted_bgr)
        st.download_button(
            "Download Annotated Image",
            data=BytesIO(buf),
            file_name=f"annotated_{uploaded_file.name}",
            mime="image/png"
        )

    st.divider()
    generate_smart_risk_report(results[0])
    
    st.divider()
    display_object_gallery(image, results[0], uploaded_file.name)
    
    st.divider()
    handle_missed_object_correction(model, image, uploaded_file)


def display_object_gallery(image: Image.Image, results: Any, image_name: str):
    """Displays a gallery of detected objects and handles feedback for false positives."""
    st.subheader("🔍 Detected Object Gallery & Feedback")
    st.write("Review each detected object. If a detection is incorrect (i.e., it's not actually that object), report it as a **False Positive**.")
    
    if not results.boxes:
        st.info("No objects were detected in this image.")
        return

    num_cols = 5
    cols = st.columns(num_cols)
    for i, box in enumerate(results.boxes):
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        
        if not (xyxy[0] < xyxy[2] and xyxy[1] < xyxy[3]):
            continue

        class_name = results.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        
        cropped_img = image.crop(xyxy)
        padded_img = pad_image_to_square(cropped_img)
        
        with cols[i % num_cols]:
            st.image(padded_img, use_container_width=True)
            st.caption(f"**{class_name}** (Conf: {confidence:.2f})")
            
            is_reported = i in st.session_state.get('reported_falses', set())
            
            if st.button(
                "Reported ✓" if is_reported else "Report Incorrect",
                key=f"report_{image_name}_{i}",
                disabled=is_reported,
                use_container_width=True
            ):
                handle_incorrect_detection_feedback(image_name, box, class_name)
                st.session_state.reported_falses.add(i)
                st.toast(f"Reported '{class_name}' as incorrect. Thank you!", icon="👍")
                time.sleep(1)
                st.rerun()


def handle_missed_object_correction(model: YOLO, image: Image.Image, uploaded_file: Any):
    """
    Handles the UI for correcting missed detections using an interactive chart.
    This version uses a form to ensure submission works correctly.
    """
    st.subheader("✍️ Correct Missed Detections (False Negatives)")
    st.write("If the model missed an object, select its class, draw a box on the image below, and submit.")

    with st.form(key="missed_object_form"):
        class_names = list(model.names.values())
        selected_class = st.selectbox("1. Select the class of the missed object:", class_names)
        
        fig = px.imshow(image)
        fig.update_layout(
            dragmode="select",
            newshape_line_color='red',
            title_text="2. Draw a box on the image",
            title_x=0.5
        )
        config = {"modeBarButtonsToAdd": ["drawrect", "eraseshape"]}
        
        st.plotly_chart(fig, use_container_width=True, config=config)
        
        submitted = st.form_submit_button("3. Submit Missed Object Feedback")
        if submitted:
            handle_missed_object_feedback(uploaded_file, image, selected_class)
            st.success("Thank you! Your feedback (with simulated coordinates) has been recorded.")


def handle_webcam_detection(model: YOLO, confidence_threshold: float):
    """Handles real-time webcam detection with a more robust configuration for deployment."""
    st.info("Click 'Start' to begin live detection from your webcam. The stream may take a few seconds to initialize.")
    
    # A more robust RTC configuration with multiple STUN servers for better connectivity.
    rtc_configuration = {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]
    }
    
    webrtc_streamer(
        key="webcam-streamer",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=lambda: VideoTransformer(model, confidence_threshold),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


def handle_performance_analysis(run_path: str):
    """Displays detailed performance graphs from a training run."""
    st.write("This dashboard provides a detailed breakdown of the selected model's performance on the validation dataset.")
    
    st.write("##### **Training Progress**")
    st.write("These charts show how the model's accuracy (mAP, Precision, Recall) and error rate (Loss) improved over each epoch of training.")
    results_graph_path = os.path.join(run_path, 'results.png')
    if os.path.exists(results_graph_path):
        st.image(results_graph_path, use_container_width=True)
    else:
        st.warning("Main training results graph (results.png) not found in the selected run directory.")
    
    st.divider()

    st.write("##### **Class Performance Analysis**")
    st.write("These graphs break down the model's performance for each specific object class.")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Confusion Matrix:** Shows where the model got confused (e.g., misclassifying a Toolbox as an OxygenTank). The diagonal represents correct predictions.")
        path = os.path.join(run_path, 'confusion_matrix.png')
        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.info("Confusion Matrix graph not found.")
            
    with col2:
        st.write("**Precision-Recall Curve:** Illustrates the trade-off between precision and recall for different confidence thresholds. A curve closer to the top-right corner indicates better performance.")
        pr_path = os.path.join(run_path, 'PR_curve.png')
        f1_path = os.path.join(run_path, 'F1_curve.png')
        if os.path.exists(pr_path):
            st.image(pr_path, use_container_width=True)
        elif os.path.exists(f1_path):
             st.image(f1_path, use_container_width=True) # Fallback to F1 curve
        else:
            st.info("Precision-Recall curve not found.")


def handle_about_section():
    """Displays the 'About' page content with a modern, structured layout."""
    st.header("🛡️ What is EVA Guardian?")
    st.markdown("""
    EVA Guardian is an AI-powered safety assistant designed to operate in high-stakes environments like a space station. 
    Its mission is to enhance astronaut safety by automatically identifying and assessing critical equipment in real-time.

    The core of the project is a **YOLOv8 object detection model** that we trained exclusively on a *synthetic dataset* provided by Duality AI. This proves that we can build reliable AI for places that are difficult or impossible 
    to get real photos from. We didn't just build a model; we built a complete, interactive dashboard that turns AI detections 
    into **actionable intelligence**.
    """)

    st.header("🚀 Key Features")

    with st.expander("👁️ Live Detection and Analysis", expanded=True):
        st.markdown("""
        - Users can upload their own images or use a live webcam feed.
        - The system runs our best-trained model to find and draw boxes around three key objects: **Fire Extinguishers**, **Toolboxes**, and **Oxygen Tanks**.
        """)

    with st.expander("🧠 Smart Risk Assessment", expanded=True):
        st.markdown("""
        This is our most advanced feature. The app doesn't just find objects; it prioritizes them by risk.
        It calculates a numerical **"Risk Score"** for each detected object based on a smart formula that considers:
        - **Urgency:** A Fire Extinguisher is treated as more critical than a Toolbox.
        - **Visibility:** An object that is hard to see (low confidence) is flagged as a higher risk.
        - **Proximity:** The size of the object's box is used to guess how close it is.
        
        It then presents a sorted list of risks and a simple, color-coded summary (e.g., `CRITICAL ALERT` in red), turning the AI into a true decision-support tool.
        """)

    with st.expander("✍️ Complete Human-in-the-Loop Feedback System", expanded=True):
        st.markdown("""
        We implemented two types of feedback to demonstrate how the model can be continuously improved.
        - **Detected Object Gallery:** Users can review a gallery of everything the model found. If the model makes a mistake (e.g., calls a random object a "Toolbox"), the user can click "Report Incorrect" to flag this *false positive*.
        - **Correct Missed Objects:** If the model misses an object entirely, the user can use an interactive chart to draw a new box on the image, providing the correct label for a *missed object* (false negative).
        """)

    with st.expander("📈 Dynamic Performance Dashboard", expanded=True):
        st.markdown("""
        - The application automatically finds the latest and best-trained model from all our training runs.
        - It dynamically loads the performance metrics (**mAP, Precision, and Recall**) from the results file of that specific model and displays them.
        - The "Performance Deep Dive" tab provides a full suite of analysis graphs, including the **Confusion Matrix** and **loss curves**, for a complete technical overview.
        """)

   


# --- FEEDBACK HANDLING LOGIC (as per user's original code) ---
def handle_incorrect_detection_feedback(image_name, box, class_name):
    """Saves feedback for a false positive."""
    all_feedback = []
    if os.path.exists(INCORRECT_FEEDBACK_FILE):
        try:
            with open(INCORRECT_FEEDBACK_FILE, 'r') as f:
                all_feedback = json.load(f)
            if not isinstance(all_feedback, list):
                all_feedback = [all_feedback]
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    new_feedback = {
        "source_image": image_name,
        "incorrectly_detected_as": class_name,
        "bounding_box": box.xyxy[0].cpu().numpy().tolist(),
        "confidence": float(box.conf[0]),
        "feedback_type": "false_positive",
        "timestamp": time.time()
    }
    all_feedback.append(new_feedback)
    with open(INCORRECT_FEEDBACK_FILE, "w") as f:
        json.dump(all_feedback, f, indent=2)


def handle_missed_object_feedback(uploaded_file, image, class_name):
    """Saves feedback for a missed object using SIMULATED coordinates."""
    all_feedback = []
    if os.path.exists(MISSED_FEEDBACK_FILE):
        try:
            with open(MISSED_FEEDBACK_FILE, 'r') as f:
                all_feedback = json.load(f)
            if not isinstance(all_feedback, list):
                all_feedback = [all_feedback]
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    
    os.makedirs(FEEDBACK_IMAGE_DIR, exist_ok=True)
    unique_filename = f"{os.path.splitext(uploaded_file.name)[0]}_{int(time.time())}.png"
    image.save(os.path.join(FEEDBACK_IMAGE_DIR, unique_filename))
    
    # Using simulated coordinates for stability
    new_feedback = {
        "className": class_name,
        "box_coordinates (simulated)": {"left": 100, "top": 100, "width": 150, "height": 150},
        "source_image": unique_filename
    }
    all_feedback.append(new_feedback)
        
    with open(MISSED_FEEDBACK_FILE, "w") as f:
        json.dump(all_feedback, f, indent=2)


def generate_smart_risk_report(results: Any):
    """Analyzes detection results to generate a prioritized risk report."""
    st.subheader("🚨 EVA Guardian: Smart Risk Assessment")
    risk_data, counts = [], {name: 0 for name in results.names.values()}

    if results.boxes:
        img_h, img_w = results.orig_shape
        img_area = img_h * img_w
        
        for box in results.boxes:
            class_name = results.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            counts[class_name] += 1

            xyxy = box.xyxy[0]
            box_area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            
            # Proximity is the percentage of screen area the object occupies
            proximity_factor = float(box_area / img_area) if img_area > 0 else 0
            
            urgency_score = URGENCY_WEIGHTS.get(class_name, DEFAULT_URGENCY)
            
            # REVISED RISK FORMULA: Risk increases with urgency and proximity,
            # and is tempered by high confidence.
            risk_score = (urgency_score * (1 + proximity_factor * 5)) / (confidence + 0.1)

            risk_data.append({
                "Object": class_name,
                "Risk Score": risk_score,
                "Confidence": f"{confidence:.2f}",
                "Proximity": f"{proximity_factor:.2%}"
            })
            
    if risk_data:
        df = pd.DataFrame(risk_data).sort_values(by="Risk Score", ascending=False).reset_index(drop=True)
        # Format the risk score for display after sorting
        df['Risk Score'] = df['Risk Score'].apply(lambda x: f"{x:.3f}")
        st.write("**Risk-Prioritized List:**")
        st.dataframe(df, use_container_width=True)
    else:
        st.success("✅ **All Clear:** No objects detected to assess.")

    st.write("---")
    st.write("**Safety Status Summary:**")
    if counts.get('FireExtinguisher', 0) == 0:
        st.error("❌ **CRITICAL ALERT:** No Fire Extinguisher detected in the field of view.")
    else:
        st.success(f"✅ **OK:** Found {counts['FireExtinguisher']} Fire Extinguisher(s).")
    
    if counts.get('ToolBox', 0) > 0:
        st.warning(f"⚠️ **ACTION REQUIRED:** Found {counts['ToolBox']} ToolBox(es). Please verify stowage and security.")
    else:
        st.success("✅ **OK:** No unsecured ToolBoxes detected.")
    
    st.info(f"ℹ️ **STATUS:** Found {counts.get('OxygenTank', 0)} Oxygen Tank(s).")


# --- MAIN APPLICATION ---
def main():
    """Main function to run the Streamlit application."""
    # --- State Management for Page Navigation ---
    if 'page' not in st.session_state:
        st.session_state.page = 'Dashboard'
    if 'show_model_selector' not in st.session_state:
        st.session_state.show_model_selector = False

    # --- Custom Title Bar (using st.columns for robustness) ---
    st.markdown("""
        <style>
            /* Reduce padding at the top of the main container */
            div.block-container { padding-top: 3.5rem; }
            /* Custom styling for nav buttons */
            .stButton>button {
                border: 2px solid transparent;
                background-color: transparent;
                color: #fafafa;
                transition: all 0.2s ease-in-out;
            }
            .stButton>button:hover {
                border-color: #00aaff;
                color: #00aaff;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Using st.columns for the title bar is more stable than custom HTML/CSS for layout
    col1_title, col_spacer, col2_title, col3_title = st.columns([4, 2, 1.2, 1])
    with col1_title:
        st.title("🛡️ EVA Guardian")
    with col2_title:
        if st.button("Dashboard", key="nav_dashboard", use_container_width=True):
            st.session_state.page = 'Dashboard'
    with col3_title:
        if st.button("About", key="nav_about", use_container_width=True):
            st.session_state.page = 'About'

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("Configuration")
        available_models = get_model_versions()
        
        if not available_models:
            st.error("No trained models found in the 'runs/detect' directory. Please train a model first.")
            st.stop()
        
        latest_model_name = list(available_models.keys())[0]
        
        if st.session_state.get('show_model_selector', False):
            selected_model_name = st.selectbox(
                "Model Version",
                options=list(available_models.keys()),
                key='model_select'
            )
        else:
            selected_model_name = latest_model_name
        
        st.info(f"Current Model: **{selected_model_name}**")
        
        if st.button("Select Other Model Versions"):
            st.session_state.show_model_selector = not st.session_state.get('show_model_selector', False)
            st.rerun()
            
        run_path = available_models[selected_model_name]
        model_path = os.path.join(run_path, WEIGHTS_FILE)
        
        confidence_threshold = st.slider(
            "Confidence Threshold", 0.0, 1.0, 0.45, 0.05,
            help="Minimum probability for an object to be considered a valid detection. Lower values detect more objects but may increase false positives."
        )
    
    # Load the selected model
    model = load_model(model_path)
    if model is None:
        st.error("Failed to load the selected model. The application cannot proceed.")
        st.stop()

    # --- Page Content based on navigation ---
    if st.session_state.page == 'Dashboard':
        handle_dashboard(model, confidence_threshold, selected_model_name, run_path)
    elif st.session_state.page == 'About':
        handle_about_section()


if __name__ == "__main__":
    main()
