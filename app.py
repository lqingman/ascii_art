import gradio as gr
import cv2
import numpy as np


def pre_process_image(image, ascii_width=130):
    """
    this function is used to preprocess the image,
    resize the image and convert it to grayscale
    """
    # Convert the image to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Split the image into its respective Red, Green, and Blue channels
    R, G, B = cv2.split(img_rgb)
    # Apply the grayscale conversion formula: 0.299*R + 0.587*G + 0.114*B
    grayscale_img = 0.299 * R + 0.587 * G + 0.114 * B
    # Ensure the pixel values are in valid range (0 to 255) and convert to uint8
    grayscale_img = grayscale_img.astype(np.uint8)
    
    # Resize the image
    height, width = grayscale_img.shape
    aspect_ratio = height / width
    new_height = int(ascii_width * aspect_ratio * 0.5)
    resized_img = cv2.resize(grayscale_img, (ascii_width, new_height))

    return resized_img


def darken_edges(contrast_img, edges, darken_factor=28):
    """
    this function is used to darken the edges of the image
    """
    # Create a copy of the contrast image to modify
    combined_image = contrast_img.copy()

    # Loop through each pixel
    for i in range(contrast_img.shape[0]):  # height
        for j in range(contrast_img.shape[1]):  # width
            # Check if the pixel is part of an edge
            if edges[i, j] > 0:  # Non-zero value means it's an edge pixel
                # Darken the pixel by reducing its intensity
                combined_image[i, j] = max(0, contrast_img[i, j] - darken_factor)

    return combined_image


def transform_grayscale(image):
    """
    this function is used to transform the grayscale image
    to make it have a better ascii representation
    """
    # Step 1: Apply CLAHE for adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_img = clahe.apply(image)

    # Step 2: Apply Median Blur instead of Gaussian Blur
    blurred_image = cv2.medianBlur(contrast_img, 5)

    # Step 3: Apply edge detection (Canny algorithm) with adjusted parameters
    edges = cv2.Canny(blurred_image, 100, 150)

    # Step 4: Darken the edges to the contrast image
    combined_img = darken_edges(contrast_img, edges)

    return combined_img


def ascii_mapping(image):
    """
    this function is used to map the grayscale image to ascii characters
    """
    # Define the ASCII characters to represent the grayscale intensity
    ascii_chars = '@%#*+=-:. '  # From dark to light

    # Initialize the ASCII representation
    ascii_art = ''
    num_chars = len(ascii_chars)
    # Map each pixel intensity to an ASCII character
    for row in image:
        for pixel in row:
            # Map pixel intensity (0-255) to the range of characters (0 to len(ascii_chars)-1)
            ascii_art += ascii_chars[int(pixel) * (num_chars - 1) // 255]
        ascii_art += "\n"

    return ascii_art


def process_image(image):
    """
    The main function to process the input image and generate ASCII art
    """
    preprocessed_img = pre_process_image(image)
    transformed_img = transform_grayscale(preprocessed_img)
    ascii_art = ascii_mapping(transformed_img)
    return ascii_art


def save_ascii_art(ascii_art):
    """
    Save the ASCII art to a text file
    """
    with open("ascii_art.txt", "w") as f:
        f.write(ascii_art)
    return "ascii_art.txt"


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            image_input = gr.Image(type="numpy", label="Upload an Image")
            examples = gr.Examples(
                examples=[
                    ["./assets/dolphin.jpg"],
                    ["./assets/man.jpg"],
                    ["./assets/bird.jpg"]
                ],
                inputs=image_input
            )
            submit_button = gr.Button("Submit")
            clear_button = gr.Button("Clear")
        with gr.Column(scale=3, min_width=300):
            ascii_output = gr.TextArea(label="ASCII Art", lines=30, elem_id="ascii-art-output")
            download_button = gr.Button("Download ASCII Art")
            download_file = gr.File(label="Download File")

    submit_button.click(fn=process_image, inputs=image_input, outputs=ascii_output)
    clear_button.click(fn=lambda: (None, ""), inputs=None, outputs=[image_input, ascii_output])
    download_button.click(fn=save_ascii_art, inputs=ascii_output, outputs=download_file)


# Add custom CSS to use a monospaced font for the Textbox
demo.css = """
#ascii-art-output {
    font-family: monospace;
    white-space: pre;

}
"""

demo.launch(share=True)
