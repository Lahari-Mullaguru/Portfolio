# License Plate Recognition

## Problem Statement

This project addresses the challenge of **License Plate Recognition**, a two-step computer vision task involving:

1. **License Plate Detection**: Identifying and locating license plates in vehicle images.
2. **Character Recognition**: Reading and decoding the alphanumeric characters on the detected license plates.

---

## Dataset Structure

The dataset consists of **three parts**:

### Training Set 1 (Vehicle Images with Plates)
- **900 images** of vehicles
- Each image has an annotated bounding box around the license plate.
- Format: `(ymin, xmin, ymax, xmax)`

### Training Set 2 (License Plate Crops with Text)
- **900 images** of individual license plates.
- Each image is labeled with the **text** on the license plate.

### Test Set
- **201 vehicle images**
- Task: 
  - Detect license plates
  - Recognize the characters in the detected plates

---

## Objectives

- Build a **model to detect license plates** from vehicle images.
- Train an **OCR (Optical Character Recognition)** system to extract alphanumeric characters.
- Evaluate performance on the **Test Set** using both detection accuracy and recognition accuracy.

---

## Libraries Used

- **Python**
- **OpenCV**
- **Tesseract OCR**
- **TensorFlow, PyTorch** 
- **Pandas, NumPy, Matplotlib, Seaborn**

---

## Steps Covered in the Project

1. **Data Loading and Exploration**
2. **Preprocessing**
   - Image resizing, normalization
   - Bounding box formatting
3. **License Plate Detection**
   - Custom object detection pipeline (using CNN, YOLO)
4. **Character Recognition**
   - Using `pytesseract` or a custom-trained CNN for OCR
5. **Model Evaluation**
   - Intersection over Union (IoU) for detection
   - Character-level accuracy for recognition

---

Make sure pytesseract is installed and configured properly to run the program.
    
