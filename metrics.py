from skimage.metrics import structural_similarity as ssim
import cv2

def calculate_ssim(original_image_path, ascii_art):
    """
    Calculate the SSIM score between the original image and its ASCII art version.
    """
    # Read the original image
    original_image = cv2.imread(original_image_path)
    
    # Convert the original image to grayscale
    original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Get the dimensions of the original image
    height, width = original_image_gray.shape
    ascii_art = cv2.imread(ascii_art)
    ascii_art = cv2.cvtColor(ascii_art, cv2.COLOR_BGR2GRAY)
    
    # Resize ASCII art image to match original image dimensions
    ascii_art_image_resized = cv2.resize(ascii_art, (width, height))
    
    # Compute SSIM
    ssim_score, _ = ssim(original_image_gray, ascii_art_image_resized, full=True)
    
    return ssim_score

# Example usage
original_image ='./assets/dolphin.jpg'
ascii_art_image = './assets/ascii_dolphin.png'
ssim_score = calculate_ssim(original_image, ascii_art_image)
print(f'dolphin SSIM Score: {ssim_score}')

original_image2 ='./assets/man.jpg'
ascii_art_image2 = './assets/ascii_man.png'
ssim_score2 = calculate_ssim(original_image2, ascii_art_image2)
print(f'man SSIM Score: {ssim_score2}')

original_image3 ='./assets/bird.jpg'
ascii_art_image3 = './assets/ascii_bird.png'
ssim_score3 = calculate_ssim(original_image3, ascii_art_image3)
print(f'bird SSIM Score: {ssim_score3}')

original_image4 ='./assets/horse.jpg'
ascii_art_image4 = './assets/ascii_horse.png'
ssim_score4 = calculate_ssim(original_image4, ascii_art_image4)
print(f'horse SSIM Score: {ssim_score4}')