from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
import os
import pandas as pd
import yaml


# Path to the directory where validation results will be saved
output_directory = '/home/ziad/Files/'

# Path to YOLO model file
model_path = '/home/ziad/best.pt'

# Path to data configuration file or classes. Default is coco128
data_path = '/home/ziad/Documents/custom_data.yaml'

# Confidence thresholds to loop through
confidence_thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Run YOLO validation for each confidence threshold
for conf_threshold in confidence_thresholds:
    # Select the device
    device = select_device('0')
    
    # Run YOLO validation with the custom confidence score
    model = YOLO(model_path)
    metrics = model.val(data=data_path, conf=conf_threshold, save=True)
    
    # Extract and save per-class metrics
    ap50_scores = metrics.box.ap50.tolist()
    ap_scores = metrics.box.ap.tolist()

    # Additional metrics
    recall = metrics.box.r.tolist()
    precision = metrics.box.p.tolist()
    f1 = metrics.box.f1.tolist()
 
    # Load class names from the data configuration file
    with open(data_path, 'r') as file:
        data_config = yaml.safe_load(file)
        class_names = data_config['names']

    # Create a dictionary to store metrics for each class
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_metrics[class_name] = {
            'MAP50': ap50_scores[i],
            'MAP50-95': ap_scores[i],
            'Recall': recall[i],
            'Precision': precision[i],
            'F1': f1[i] if len(f1) > i else None
        }

    # Save the metrics for each class to a CSV file
    metrics_df = pd.DataFrame(class_metrics).transpose()
    csv_path = os.path.join(output_directory, f'validation_results_{conf_threshold}.csv')
    metrics_df.to_csv(csv_path, index_label='Class')

    print(f"Results for confidence threshold {conf_threshold} saved to: {csv_path}")
