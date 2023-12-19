# License Plate Detection and OCR Application

This application uses a trained FCOS model for license plate detection and EasyOCR for optical character recognition (OCR) to read text from detected license plates. It provides a user-friendly graphical interface to upload images, view detected license plates, and display the recognized text.

## Features
- License plate detection using a pre-trained FCOS model and fined tune with license plate dataset.
- Text recognition from detected license plates using EasyOCR.
- GUI for easy interaction, image uploading, and result display.

## Installation

### Prerequisites
- Python 3.x
- Pip (Python package manager)

### Libraries
This application requires the following libraries:
- OpenCV
- EasyOCR
- PyTorch
- TorchVision
- PIL (Python Imaging Library, also known as Pillow)
- Tkinter (usually comes pre-installed with Python)

### Steps
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Dalek12/License_Plate_Detection.git
   cd License_Plate_Detection
2. **Install essential libraries**
   ```bash
   pip install opencv-python-headless easyocr torch torchvision pillow
3. **Model path**
   Keep fcos_license_plate_detector.pth in the same directory as the fcos_ocr.py
3. **Run the APP**
   ```bash
   python fcos_ocr.py
## Dataset

The model was trained and fine-tuned using the "vehicle and license plate Computer Vision Project" dataset, provided by a Roboflow user and licensed under the MIT.

### Dataset Details
- **Dataset Name**: vehicle and license plate Computer Vision Project
- **Provided by**: [Roboflow](https://universe.roboflow.com/plat-kendaraan/vehicle-and-license-plate)
- **License**: MIT
- **Completion and Credits**: The project was completed by two contributors:
1. Irvan Teady Sentosa
   - URL: [Irvan Teady Sentosa - Indonesian License Plate Dataset](https://universe.roboflow.com/irvan-teady-sentosa-fqt01/indonesian-license-plate-o3tgv)
   - Published on Roboflow Universe, a platform for sharing and accessing datasets.
   - Visited on: 19th June 2023.

2. Habib Robbani
   - URL: [Habib Robbani - Indonesian Motorcycle Plate Dataset](https://universe.roboflow.com/habib-robbani/indonesian-motorcycle-plate)
   - Published on Roboflow Universe.
   - Visited on: 19th June 2023.
