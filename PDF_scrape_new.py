import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load SmolDocling model & processor
model_name = "ds4sd/SmolDocling-256M-preview"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForImageTextToText.from_pretrained(model_name)

# Define folders
input_folder = r"C:\Users\u0106491\Documents\LSTAT\Masterthesissen\Voorstellen JFK\1. Extract information from web\jfk_2025_pdfs_reduced"
output_folder = r"C:\Users\u0106491\Documents\LSTAT\Masterthesissen\Voorstellen JFK\1. Extract information from web\jfk_2025_texts_reduced"
os.makedirs(output_folder, exist_ok=True)

def preprocess_image(image):
    """ Convert image to grayscale, increase contrast, and apply denoising """
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)  # Remove noise
    _, thresh = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Binarization
    return Image.fromarray(thresh)

def process_image_with_smoldocling(image):
    """ Process the image with SmolDocling to extract structured text """
    inputs = processor(images=image, return_tensors="pt")
    
    if "pixel_values" not in inputs:
        print("Error: 'pixel_values' not found in inputs!")
        return ""
    
    try:
        outputs = model.generate(pixel_values=inputs["pixel_values"])
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return generated_text
    except Exception as e:
        print(f"Error during model inference: {e}")
        return ""

def clean_text(text):
    """ Post-processing to improve text quality """
    text = text.replace("-\n", "")  # Remove hyphenated line breaks
    text = text.replace("\n", " ")  # Convert new lines to spaces
    text = " ".join(text.split())  # Remove extra spaces
    return text

# Process each PDF
for pdf_file in os.listdir(input_folder):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(input_folder, pdf_file)
        images = convert_from_path(pdf_path)

        extracted_text = []
        for img in images:
            preprocessed_img = preprocess_image(img)

            # Get OCR text
            raw_text = pytesseract.image_to_string(preprocessed_img, lang="eng")
            raw_text = clean_text(raw_text)

            # Process with SmolDocling
            structured_text = process_image_with_smoldocling(preprocessed_img)
            structured_text = clean_text(structured_text)

            extracted_text.append(f"OCR Text:\n{raw_text}\n\nSmolDocling Output:\n{structured_text}")

        # Save output
        output_path = os.path.join(output_folder, f"{pdf_file[:-4]}_processed.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(extracted_text))

        print(f"Processed: {pdf_file} -> {output_path}")

print("Enhanced text extraction complete!")