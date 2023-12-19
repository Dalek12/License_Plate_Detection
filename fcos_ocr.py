import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from PIL import ImageDraw
import cv2
import numpy as np
import easyocr
import torch
import torchvision
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
from torchvision.transforms import functional as F

reader = easyocr.Reader(['en'])
root = tk.Tk()

# Load the trained model
def load_trained_model(model_path):
    weights = FCOS_ResNet50_FPN_Weights.DEFAULT
    model = fcos_resnet50_fpn(weights=weights)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
#Below are the functions to detect and extract text from plates
def extract_text_from_plate(plate_image):
    results = reader.readtext(plate_image)
    threshold = 0.5  # Threshold for confidence score

    # Return empty string if no results
    if not results:
        return ""

    # Filter results based on confidence threshold
    filtered_results = [result[1] for result in results if result[2] >= threshold]

    # Fallback if no results meet the threshold
    if not filtered_results:
        # Option 1: Return text with the highest confidence
        highest_confidence_text = max(results, key=lambda x: x[2])[1]
        return highest_confidence_text
        # Option 2: Handle as an error or request manual review
        # return "Text below confidence threshold"

    return ' '.join(filtered_results)

def detect_license_plate(model, image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)

    plates = []
    boxes = []
    for i in range(len(predictions[0]['boxes'])):
        box = predictions[0]['boxes'][i].cpu().numpy()
        score = predictions[0]['scores'][i].cpu().numpy()

        if score > 0.5:  # Threshold
            x1, y1, x2, y2 = map(int, box)
            cropped_plate = image_rgb[y1:y2, x1:x2]
            plates.append(cropped_plate)
            boxes.append((x1, y1, x2, y2))

    return plates, boxes
def display_image_with_boxes(image_path, boxes):
    # Open the image and create a drawing context
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Draw bounding boxes
    for box in boxes:
        draw.rectangle(box, outline="red", width=2)

    # Resize for display if needed
    image.thumbnail((800, 600))
    photo = ImageTk.PhotoImage(image)

    # Display the image
    image_label.config(image=photo)
    image_label.image = photo
# Function to open an image file
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load and display the image
        image = Image.open(file_path)
        image.thumbnail((500, 500))  # Resize for display
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo

        # Process the image for license plate detection and OCR
        process_image(file_path)

# Function to process the image and display results
def process_image(file_path):
    # Your detection and OCR code goes here
    model = load_trained_model('fcos_license_plate_detector.pth')
    plates, boxes = detect_license_plate(model, file_path)
    detected_text = ''
    for plate_image in plates:
        if isinstance(plate_image, np.ndarray) and plate_image.size != 0:
            text = extract_text_from_plate(plate_image)
            detected_text = text
        else:
            print("No valid plate image found.")

    # Display the results
    display_image_with_boxes(file_path, boxes)
    result_text.set("Detected text: " + detected_text)
    result_label.delete('1.0', tk.END)
    result_label.insert(tk.END, "Detected text: " + detected_text)


root.title("License Plate Detection and OCR")
root.geometry("800x600") 

# Frame for buttons and text
top_frame = tk.Frame(root)
top_frame.pack(side=tk.TOP, fill=tk.X)

# Frame for image display
image_frame = tk.Frame(root)
image_frame.pack(fill=tk.BOTH, expand=True)

# Setup GUI elements in the top frame
open_button = tk.Button(top_frame, text="Open Image", command=open_image)
open_button.pack(side=tk.LEFT, padx=10, pady=10)

# Scrollable text display in the top frame
text_scroll = tk.Scrollbar(top_frame)
text_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

result_text = tk.StringVar()
result_label = tk.Text(top_frame, yscrollcommand=text_scroll.set, height=4)
result_label.pack(fill=tk.X, expand=True)
text_scroll.config(command=result_label.yview)

# Image label in the image frame
image_label = tk.Label(image_frame)
image_label.pack(fill=tk.BOTH, expand=True, pady=10)

# Run the application
root.mainloop()