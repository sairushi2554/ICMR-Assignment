import cv2
import os
from PIL import Image, ImageEnhance

class ImageProcessor:
    """
    A class to handle image processing operations for text recognition
    """
    
    def __init__(self):
        """Initialize the ImageProcessor"""
        pass
        
    def enhance_image(self, image_path):
        """
        Enhance the image to improve text recognition
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Path to the preprocessed image
        """
        # Open image
        img = Image.open(image_path)
        print(f"[DEBUG] Original image mode: {img.mode}")
        
        # Ensure image is in RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
            print(f"[DEBUG] Converted to RGB: {img.mode}")
            
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)
        
        # Save preprocessed image in the same directory as the original
        output_dir = os.path.dirname(image_path)
        basename = os.path.basename(image_path)
        preprocessed_path = os.path.join(output_dir, f"preprocessed_{basename}")
        
        img.save(preprocessed_path)
        print(f"[DEBUG] Final saved image mode: {img.mode}")
        print(f"[DEBUG] Preprocessed image saved to: {preprocessed_path}")
        
        return preprocessed_path
    
    def visualize_results(self, image_path, results):
        """
        Visualize the detected text regions on the image
        
        Args:
            image_path: Path to the image
            results: OCR detection results
        """
        import matplotlib.pyplot as plt
        
        # Read image
        image = cv2.imread(image_path)
        
        # Draw bounding boxes and text
        for (bbox, text, prob) in results:
            # Convert bbox to integer points
            tl = tuple(map(int, bbox[0]))  # top left
            br = tuple(map(int, bbox[2]))  # bottom right
            
            # Draw rectangle
            cv2.rectangle(image, tl, br, (0, 255, 0), 2)
            
            # Put text
            cv2.putText(image, text, (tl[0], tl[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Convert from BGR to RGB for matplotlib
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Show image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.title("Detected Handwritten Text")
        plt.axis('off')
        plt.show()