# PDF Annotation Merger

This project is designed to merge annotations from two separate annotated PDFs onto an original PDF. It helps users combine multiple annotations (for instance, from different people or versions) onto a single original document, creating a final merged version with all the required annotations.

## Features

- Convert PDF slides to images for comparison.
- Detect differences between two annotated PDFs and the original.
- Remove overlapping annotations based on bounding box expansion.
- Combine annotations from two PDFs onto the original and save the result as a new PDF.
- Supports working with high-resolution PDF images (300 DPI).

## Setup

1. Clone this repository or download the script.
2. Install the required dependencies:
   ```bash
   pip install opencv-python numpy pdf2image fpdf

Ensure you have poppler-utils installed on your system (required by pdf2image for PDF to image conversion).
- On Ubuntu:
    ```bash
    sudo apt-get install poppler-utils
- On macOS (Homebrew):
    ```bash
    brew install poppler

## How to run

1. Prepare the following PDFs:
    - original_pdf: The original, unannotated PDF.
    - annotated_pdf1: The first annotated PDF.
    - annotated_pdf2: The second annotated PDF.

2. Run the script from the command line:
    ```bash
    python merge_annotations.py <original_pdf> <annotated_pdf1> <annotated_pdf2> <output_pdf>

## Example Use Case

Let's say you have a document that needs to be reviewed by two people. Each reviewer annotates the document in their own way. 
This script can be used to merge both sets of annotations onto the original PDF so that you end up with a combined version for easy review.

## Future Work

1. Possiblity to have git style merge conflict resolution
2. GUI