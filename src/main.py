import cv2
import numpy as np
from pdf2image import convert_from_path
from fpdf import FPDF
import os
import argparse

def pdf_to_images(pdf_path, dpi=300):
    """Convert a PDF to a list of images."""
    images = convert_from_path(pdf_path, dpi=dpi)
    return [np.array(img) for img in images]

def find_differences_enhanced(original, annotated):
    """Find differences between original and annotated images."""
    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_original, gray_annotated)
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    return mask

def find_contours(mask):
    """Find contours in a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def expand_bounding_box(bbox, img_shape, expansion=10):
    """Expand a bounding box with a given expansion factor."""
    x, y, w, h = bbox
    new_x = max(0, x - expansion)
    new_y = max(0, y - expansion)
    new_w = min(img_shape[1], x + w + expansion) - new_x
    new_h = min(img_shape[0], y + h + expansion) - new_y
    return new_x, new_y, new_w, new_h

def remove_overlapping_annotations(mask1, mask2):
    """Remove overlapping annotations from mask2 based on mask1."""
    contours1 = find_contours(mask1)
    contours2 = find_contours(mask2)

    mask2_modified = mask2.copy()
    for contour1 in contours1:
        bbox = cv2.boundingRect(contour1)
        expanded_bbox = expand_bounding_box(bbox, mask1.shape)
        x, y, w, h = expanded_bbox

        region_of_influence = np.zeros_like(mask1, dtype=np.uint8)
        cv2.rectangle(region_of_influence, (x, y), (x + w, y + h), 255, -1)

        for contour2 in contours2:
            temp_mask = np.zeros_like(mask2, dtype=np.uint8)
            cv2.drawContours(temp_mask, [contour2], -1, 255, -1)
            overlap = cv2.bitwise_and(temp_mask, region_of_influence)

            if np.any(overlap):
                cv2.drawContours(mask2_modified, [contour2], -1, 0, -1)

    return mask2_modified

def apply_annotations_from_two_images(original, annotated1, annotated2, mask1, mask2):
    """Merge annotations from two annotated images onto the original image."""
    mask1 = np.where(mask1 > 0, 255, 0).astype(np.uint8)
    mask2 = np.where(mask2 > 0, 255, 0).astype(np.uint8)

    annotations1 = cv2.bitwise_and(annotated1, annotated1, mask=mask1)
    annotations2 = cv2.bitwise_and(annotated2, annotated2, mask=mask2)

    combined_mask = cv2.bitwise_or(mask1, mask2)
    masked_original = cv2.bitwise_and(original, original, mask=cv2.bitwise_not(combined_mask))

    combined = cv2.add(masked_original, annotations1)
    combined = cv2.add(combined, annotations2)

    return combined

def merge_annotations(original_pdf, annotated_pdf1, annotated_pdf2, output_pdf, dpi=300):
    """Merge annotations for all slides and save the final output as a PDF."""
    original_images = pdf_to_images(original_pdf, dpi=dpi)
    annotated_images1 = pdf_to_images(annotated_pdf1, dpi=dpi)
    annotated_images2 = pdf_to_images(annotated_pdf2, dpi=dpi)

    pdf = FPDF()

    for i, (original, annotated1, annotated2) in enumerate(zip(original_images, annotated_images1, annotated_images2)):
        mask1 = find_differences_enhanced(original, annotated1)
        mask2 = find_differences_enhanced(original, annotated2)

        mask2_modified = remove_overlapping_annotations(mask1, mask2)

        final_image = apply_annotations_from_two_images(original, annotated1, annotated2, mask1, mask2_modified)

        # Convert final image to RGB for PDF compatibility
        final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

        # Save directly to the PDF
        temp_image_path = f"temp_slide_{i + 1}.jpg"
        cv2.imwrite(temp_image_path, final_image_rgb)
        pdf.add_page()
        pdf.image(temp_image_path, x=0, y=0, w=210, h=297)  # A4 size (210x297 mm)

        # Clean up temporary image
        os.remove(temp_image_path)

    # Save the final merged PDF
    pdf.output(output_pdf)
    print(f"Merged PDF saved as: {output_pdf}")

def main():
    parser = argparse.ArgumentParser(description="Merge annotations from two PDFs onto the original PDF.")
    parser.add_argument("original_pdf", type=str, help="Path to the original PDF.")
    parser.add_argument("annotated_pdf1", type=str, help="Path to the first annotated PDF.")
    parser.add_argument("annotated_pdf2", type=str, help="Path to the second annotated PDF.")
    parser.add_argument("output_pdf", type=str, help="Path to the output merged PDF.")
    args = parser.parse_args()

    merge_annotations(
        args.original_pdf,
        args.annotated_pdf1,
        args.annotated_pdf2,
        args.output_pdf
    )

if __name__ == "__main__":
    main()
