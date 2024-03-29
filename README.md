# YOLO Validation for All Classes

This repository contains Python code for performing YOLO object detection validation across various confidence thresholds for all classes and outputing the results in excel file. The script evaluates a YOLO model's performance on a custom dataset, computing metrics such as Mean Average Precision (MAP) and other metrics for each class at different confidence thresholds.

## Overview

Object detection models, like YOLO (You Only Look Once), are widely used in computer vision tasks to detect objects within images. Evaluating the performance of such models is crucial for assessing their accuracy and reliability in real-world scenarios. This repository provides a convenient way to perform validation for YOLO models across different confidence thresholds, enabling thorough analysis of model performance.

## Features

- Supports evaluation of YOLO models on custom datasets.
- Computes precision, recall, MAP, and other metrics for each class.
- Iterates through a range of confidence thresholds for comprehensive evaluation.
- Saves evaluation results for each confidence threshold to CSV files for further analysis.
- Utilizes Ultralytics library for YOLO model evaluation.

## Usage

1. Clone this repository to your local machine.
2. Ensure you have Python installed along with the required dependencies listed below.
3. Update the paths to the YOLO model file (`best.pt`), data configuration file (`custom_data.yaml`), and output directory (`Files/`) in the script.
4. Run the script `yolo_val_for_all_classes.py`.
5. View the generated CSV files containing evaluation results for each confidence threshold.

## Requirements

- Python 3.x
- Ultralytics library
- PyYAML
- Pandas

## Credits

This code is inspired by the [Ultralytics YOLO](https://github.com/ultralytics/) repository and extends its functionality to perform validation for all classes across different confidence thresholds.

## License

This project is licensed under the [GNU 3 License](LICENSE).
