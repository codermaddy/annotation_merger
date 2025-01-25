import cv2
import numpy as np
from pdf2image import convert_from_path

def pdf_to_images(pdf_path, dpi=300):
    images = convert_from_path(pdf_path, dpi=dpi)
    return [np.array(img) for img in images]

def find_differences_enhanced(original, annotated):
    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_original, gray_annotated)
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    return mask

def find_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def expand_bounding_box(bbox, img_shape, expansion=10):
    x, y, w, h = bbox
    new_x = max(0, x - expansion)
    new_y = max(0, y - expansion)
    new_w = min(img_shape[1], x + w + expansion) - new_x
    new_h = min(img_shape[0], y + h + expansion) - new_y
    return new_x, new_y, new_w, new_h

def remove_overlapping_annotations(mask1, mask2):
    contours1 = find_contours(mask1)
    contours2 = find_contours(mask2)

    mask2_modified = mask2.copy()
    for contour1 in contours1:
        bbox = cv2.boundingRect(contour1)
        expanded_bbox = expand_bounding_box(bbox, mask1.shape)
        x, y, w, h = expanded_bbox
        
        # Create a region of influence from mask1
        region_of_influence = np.zeros_like(mask1, dtype=np.uint8)
        cv2.rectangle(region_of_influence, (x, y), (x + w, y + h), 255, -1)
        
        for contour2 in contours2:
            temp_mask = np.zeros_like(mask2, dtype=np.uint8)
            cv2.drawContours(temp_mask, [contour2], -1, 255, -1)
            overlap = cv2.bitwise_and(temp_mask, region_of_influence)
            
            if np.any(overlap):
                # Remove this contour from mask2
                cv2.drawContours(mask2_modified, [contour2], -1, 0, -1)

    return mask2_modified

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
annotated_slide1 = "test_files/annotations-1.pdf"
annotated_slide2 = "test_files/annotations-2.pdf"

# Step 1: Convert PDFs to images
original_img = pdf_to_images(original_slide, dpi=300)[5]
annotated_img1 = pdf_to_images(annotated_slide1, dpi=300)[6]
annotated_img2 = pdf_to_images(annotated_slide2, dpi=300)[5]

# Step 2: Create binary masks for annotations
mask1 = find_differences_enhanced(original_img, annotated_img1)
mask2 = find_differences_enhanced(original_img, annotated_img2)

# Step 3: Remove overlapping annotations from mask2
mask2_modified = remove_overlapping_annotations(mask1, mask2)

# Save intermediate results
cv2.imwrite("mask1.png", mask1)
cv2.imwrite("mask2.png", mask2)
cv2.imwrite("mask2_modified.png", mask2_modified)

# Step 4: Apply annotations to the original image
final_image = apply_annotations_from_two_images(original_img, annotated_img1, annotated_img2, mask1, mask2_modified)

# Save final merged image
cv2.imwrite("merged_annotations_area_based.png", final_image)
