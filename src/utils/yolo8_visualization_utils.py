### Modified visualization_utils.py to support YOLOv8 ###

import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_detections(image, detections, class_names, img_size=(640, 640)):
    """
    Plot detections on the input image.

    Args:
        image: The input image (numpy array).
        detections: Detected bounding boxes with format [center_x, center_y, width, height, class_scores].
        class_names: List of class names.
        img_size: The size of the image.
    """
    img = cv2.resize(image, img_size)
    for det in detections:
        center_x, center_y, width, height = det[:4]
        class_scores = det[4:]
        cls = np.argmax(class_scores)
        confidence = class_scores[cls]

        # Convert normalized center coordinates to pixel coordinates
        x1 = int((center_x - width / 2) * img_size[0])
        y1 = int((center_y - height / 2) * img_size[1])
        x2 = int((center_x + width / 2) * img_size[0])
        y2 = int((center_y + height / 2) * img_size[1])

        # Draw bounding box
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cls]}: {confidence:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def save_detection_image(image, detections, class_names, save_path, img_size=(640, 640)):
    """
    Save the image with plotted detections.

    Args:
        image: The input image (numpy array).
        detections: Detected bounding boxes with format [center_x, center_y, width, height, class_scores].
        class_names: List of class names.
        save_path: Path to save the image.
        img_size: The size of the image.
    """
    img = cv2.resize(image, img_size)
    for det in detections:
        center_x, center_y, width, height = det[:4]
        class_scores = det[4:]
        cls = np.argmax(class_scores)
        confidence = class_scores[cls]

        # Convert normalized center coordinates to pixel coordinates
        x1 = int((center_x - width / 2) * img_size[0])
        y1 = int((center_y - height / 2) * img_size[1])
        x2 = int((center_x + width / 2) * img_size[0])
        y2 = int((center_y + height / 2) * img_size[1])

        # Draw bounding box
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cls]}: {confidence:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(save_path, img)