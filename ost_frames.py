# Input-> Directory(containing frames)
# Output-> Extracted text(.xlsx)
# In this script we are extracting text from the frames by masking the frame and highlighting the text

import os
import cv2
import easyocr
import numpy as np
from openpyxl import Workbook
import warnings

warnings.filterwarnings("ignore")

# Initialize EasyOCR reader once
reader = easyocr.Reader(['en'], gpu=False)

def preprocess_image(image_path):
    """
    Preprocesses an image for better OCR detection.
    Includes grayscale, resizing, and CLAHE for contrast enhancement.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None # Return None for both preprocessed_img and original_img

    original_img_copy = img.copy() # Keep a copy of the original for masking/saving

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # You can experiment with these values or comment out if they worsen results
    resized = cv2.resize(gray, (0, 0), fx=0.7, fy=0.7) # Your original resizing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)

    return enhanced, original_img_copy # Return preprocessed_img and original_img

def detect_and_mask_text(preprocessed_img, original_img, min_text_len=1, confidence_threshold=0.3):
    """
    Detects text using EasyOCR on the preprocessed image.
    Generates a masked image where only detected text regions are visible.
    Returns the detected text string and the masked image.
    """
    if preprocessed_img is None or original_img is None:
        return "", None

    # Perform OCR on the preprocessed image
    results = reader.readtext(preprocessed_img)

    # Initialize a black mask the same size as the original image (for visual output)
    mask = np.zeros_like(original_img)
    # List to store detected text for output
    detected_text_list = []

    for box, text, conf in results:
        # Filter based on confidence and minimum text length
        if conf >= confidence_threshold and len(text.strip()) >= min_text_len:
            detected_text_list.append(text.strip())

            pts = np.array(box).astype(np.int32)
            if original_img.shape[0] > preprocessed_img.shape[0]: # If original is larger
                scale_x = original_img.shape[1] / preprocessed_img.shape[1]
                scale_y = original_img.shape[0] / preprocessed_img.shape[0]
                pts = (pts * np.array([scale_x, scale_y])).astype(np.int32)

            cv2.fillPoly(mask, [pts], (255, 255, 255)) # White text on black mask

    # Apply the mask to the original image
    masked_output_img = cv2.bitwise_and(original_img, mask)

    return " ".join(detected_text_list).strip(), masked_output_img


def extract_text_to_excel(input_folder, output_excel="frame_text_data.xlsx", output_folder="frames_text_only"):
    os.makedirs(output_folder, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "Text Frames"
    ws.append(["Frame Name", "Detected Text"])

    total, matched = 0, 0

    print(f"Starting text extraction from: {input_folder}")
    print(f"Output Excel: {output_excel}")
    print(f"Output masked frames: {output_folder}\n")

    for file in sorted(os.listdir(input_folder)):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        total += 1
        image_path = os.path.join(input_folder, file)

        try:
            # 1. Preprocess the image and get a copy of the original
            preprocessed_img, original_img = preprocess_image(image_path)
            if preprocessed_img is None:
                print(f"Skipping {file}: Could not read image.")
                continue

            # 2. Detect text and generate the masked image in one go
            detected_text, masked_image_for_saving = detect_and_mask_text(
                preprocessed_img, original_img,
                min_text_len=1, confidence_threshold=0.3 # Use consistent thresholds
            )

            if detected_text:
                ws.append([file, detected_text])
                out_path = os.path.join(output_folder, file)
                cv2.imwrite(out_path, masked_image_for_saving) # Save the masked image
                print(f"Text detected in: {file}:  Text: '{detected_text}'")
                matched += 1
            else:
                print(f"{file}: No significant text detected.")

        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    wb.save(output_excel)
    print(f"\n--- Process Summary ---")
    print(f"Excel saved: {output_excel}")
    print(f"{matched} frames with text saved to: {output_folder}")
    print(f"Total frames scanned: {total}")

if __name__ == "__main__":
    folder = input("Enter path to folder with frames: ").strip()
    if not os.path.isdir(folder):
        print(f"Error: Folder '{folder}' does not exist or is not a directory.")
    else:
        extract_text_to_excel(folder)
