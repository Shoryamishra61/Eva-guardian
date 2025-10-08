import json
import os
import shutil
from PIL import Image
import yaml 
import datetime

def log(message):
    """Logs a message to the console with a timestamp."""
    now = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {message}")

def get_class_map_from_yaml(yaml_path):
    """Dynamically creates the class map from the yolo_params.yaml file."""
    try:
        with open(yaml_path, 'r') as f:
            params = yaml.safe_load(f)
        if 'names' in params and isinstance(params['names'], list):
            return {name: i for i, name in enumerate(params['names'])}
        else:
            log("Error: 'names' key not found or not a list in yolo_params.yaml.")
            return None
    except Exception as e:
        log(f"Error reading or parsing YAML file: {e}")
        return None

def convert_to_yolo_format(box_data, image_width, image_height, class_map):
    """Converts bounding box data to YOLO's normalized format."""
    try:
        class_name = box_data['className']
        box = box_data['box_coordinates (simulated)']
        x0, y0, x1, y1 = box['x0'], box['y0'], box['x1'], box['y1']

        box_width = x1 - x0
        box_height = y1 - y0
        center_x = x0 + (box_width / 2)
        center_y = y0 + (box_height / 2)

        norm_center_x = center_x / image_width
        norm_center_y = center_y / image_height
        norm_width = box_width / image_width
        norm_height = box_height / image_height
        
        class_index = class_map.get(class_name)

        if class_index is None:
            log(f"Error: Class name '{class_name}' not found in class_map.")
            return None

        return f"{class_index} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
    except KeyError as e:
        log(f"Error processing box data. Missing key: {e}")
        return None

def process_missed_objects(class_map):
    """Processes feedback for objects the model missed."""
    log("\n--- Processing Feedback for MISSED Objects (from feedback.json) ---")
    feedback_file = 'feedback.json'
    feedback_image_dir = 'new_user_images'
    train_images_path = os.path.join('data', 'train', 'images')
    train_labels_path = os.path.join('data', 'train', 'labels')

    if not os.path.exists(feedback_file):
        log("No new `feedback.json` file found. Nothing to process.")
        return

    try:
        with open(feedback_file, 'r') as f: all_feedback = json.load(f)
    except json.JSONDecodeError:
        log(f"Error: Could not decode {feedback_file}. It might be empty or corrupt.")
        return
        
    if not isinstance(all_feedback, list):
        log(f"Error: {feedback_file} is not a list of feedback entries.")
        return
        
    log(f"Found {len(all_feedback)} entries for missed objects.")
    for feedback_data in all_feedback:
        try:
            image_name = feedback_data['source_image']
            source_image_path = os.path.join(feedback_image_dir, image_name)
            
            if not os.path.exists(source_image_path):
                log(f"SKIPPING: Source image '{image_name}' not found.")
                continue

            destination_image_path = os.path.join(train_images_path, image_name)
            shutil.copy(source_image_path, destination_image_path)
            
            with Image.open(destination_image_path) as img:
                width, height = img.size

            yolo_label_str = convert_to_yolo_format(feedback_data, width, height, class_map)

            if yolo_label_str:
                label_filename = os.path.splitext(image_name)[0] + '.txt'
                label_path = os.path.join(train_labels_path, label_filename)
                with open(label_path, 'a') as f: f.write(yolo_label_str + '\n') 
                log(f"Successfully appended new label to: {label_path}")
        except Exception as e:
            log(f"SKIPPING: Error processing entry: {e}")
    
    # Archive the processed file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.rename(feedback_file, f"feedback_processed_{timestamp}.json")
    log(f"Archived {feedback_file}.")

def process_incorrect_detections():
    """Processes feedback for objects the model detected incorrectly."""
    log("\n--- Processing Feedback for INCORRECT Detections (from incorrect_feedback.json) ---")
    feedback_file = 'incorrect_feedback.json'
    train_labels_path = os.path.join('data', 'train', 'labels')

    if not os.path.exists(feedback_file):
        log("No new `incorrect_feedback.json` file found. Nothing to process.")
        return

    try:
        with open(feedback_file, 'r') as f: all_feedback = json.load(f)
    except json.JSONDecodeError:
        log(f"Error: Could not decode {feedback_file}. It might be empty or corrupt.")
        return

    if not isinstance(all_feedback, list):
        log(f"Error: {feedback_file} is not a list of feedback entries.")
        return
        
    log(f"Found {len(all_feedback)} entries for incorrect detections.")
    for feedback_data in all_feedback:
        try:
            image_name = feedback_data['source_image']
            label_filename = os.path.splitext(image_name)[0] + '.txt'
            label_path = os.path.join(train_labels_path, label_filename)

            if os.path.exists(label_path):
                # By creating an empty file, we are telling YOLO there are no objects
                # in this image, teaching it what NOT to detect.
                open(label_path, 'w').close()
                log(f"Marked as negative example by creating empty label: {label_path}")
            else:
                log(f"Note: Label file for {image_name} did not exist, but creating empty one anyway.")
                open(label_path, 'w').close()
        except Exception as e:
            log(f"SKIPPING: Error processing entry: {e}")

    # Archive the processed file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.rename(feedback_file, f"incorrect_feedback_processed_{timestamp}.json")
    log(f"Archived {feedback_file}.")

def main():
    log("--- Starting Feedback Processing Script ---")
    yolo_params_path = 'yolo_params.yaml'
    
    class_map = get_class_map_from_yaml(yolo_params_path)
    if not class_map:
        log("Could not load class map. Aborting.")
        return
    
    # Process both types of feedback
    process_missed_objects(class_map)
    process_incorrect_detections()
    
    log("\n--- Feedback processing complete. ---")

if __name__ == '__main__':
    main()
