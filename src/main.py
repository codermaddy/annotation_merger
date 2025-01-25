import cv2
import numpy as np
from pdf2image import convert_from_path

def pdf_to_images(pdf_path, dpi=300):
    images = convert_from_path(pdf_path, dpi=dpi)
    return [np.array(img) for img in images]

def enhance_contrast(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast using histogram equalization
    enhanced = cv2.equalizeHist(gray)
    
    return enhanced

def find_differences_enhanced(original, annotated):
    # Enhance contrast before finding differences
    original_enhanced = enhance_contrast(original)
    annotated_enhanced = enhance_contrast(annotated)

    # Compute absolute difference
    diff = cv2.absdiff(original_enhanced, annotated_enhanced)

    # Apply a threshold to convert to binary mask (0 and 255)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)  # Adjust threshold here

    # Force binary mask (ensure all non-zero values are set to 255)
    thresh = np.where(thresh > 0, 255, 0).astype(np.uint8)  # Convert to binary (0, 255)

    # Optional: Clean noise using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return cleaned

def apply_annotations_from_two_images(original, annotated1, annotated2, mask1, mask2):
    # Ensure both masks are binary (0 for background, 255 for annotations)
    mask1 = np.where(mask1 > 0, 255, 0).astype(np.uint8)
    mask2 = np.where(mask2 > 0, 255, 0).astype(np.uint8)

    # Extract annotations from both annotated images using their respective masks
    annotations1 = cv2.bitwise_and(annotated1, annotated1, mask=mask1)
    annotations2 = cv2.bitwise_and(annotated2, annotated2, mask=mask2)

    # Combine the masks from both annotated images (logical OR)
    combined_mask = cv2.bitwise_or(mask1, mask2)

    # Mask out the original image where there are no annotations
    masked_original = cv2.bitwise_and(original, original, mask=cv2.bitwise_not(combined_mask))

    # Combine the original image with both sets of annotations
    combined = cv2.add(masked_original, annotations1)
    combined = cv2.add(combined, annotations2)

    return combined


# Paths to files
original_slide = "test_files/original.pdf"
annotated_slide_1 = "test_files/annotations-1.pdf"
annotated_slide_2 = "test_files/annotations-2.pdf"

# Step 1: Convert PDFs to images
original_img = pdf_to_images(original_slide, dpi=300)[5]
annotated_img_1 = pdf_to_images(annotated_slide_1, dpi=300)[6]
annotated_img_2 = pdf_to_images(annotated_slide_2, dpi=300)[5]

# Save original and annotated images
cv2.imwrite("original_image.png", original_img)  # Save original image
cv2.imwrite("annotated_image_1.png", annotated_img_1)  # Save annotated image 1
cv2.imwrite("annotated_image_2.png", annotated_img_2)  # Save annotated image 2

# Step 2: Find differences with enhanced contrast and thresholding
mask_1 = find_differences_enhanced(original_img, annotated_img_1)
mask_2 = find_differences_enhanced(original_img, annotated_img_2)

cv2.imwrite("mask_1.png", mask_1)
cv2.imwrite("mask_2.png", mask_2)

# Step 3: Combine the differences from both annotated slides
combined_mask = cv2.bitwise_or(mask_1, mask_2)

# Save the combined mask
cv2.imwrite("combined_mask.png", combined_mask)  # Save combined mask

combined_mask_colored = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
cv2.imwrite("combined_mask_colored.png", combined_mask_colored)  # For visibility

# Apply the annotations from both annotated images
merged_image = apply_annotations_from_two_images(original_img, annotated_img_1, annotated_img_2, mask_1, mask_2)

# Save the final merged image
cv2.imwrite("merged_with_annotations_from_both.png", merged_image)