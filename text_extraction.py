import easyocr
import cv2
from image_processor import ImageProcessor

class TextRecognizer:
    """
    A class to handle handwritten text recognition using EasyOCR
    """
    
    def __init__(self, languages=['en']):
        """
        Initialize the TextRecognizer
        
        Args:
            languages: List of language codes (default: English)
        """
        self.languages = languages
        self.image_processor = ImageProcessor()
        self.has_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.reader = easyocr.Reader(languages, gpu=self.has_gpu)
    
    def recognize_text(self, image_path):
        """
        Extract handwritten text from an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple containing (extracted_text, results, preprocessed_path)
        """
        try:
            # Enhance image for better recognition
            preprocessed_path = self.image_processor.enhance_image(image_path)
            
            # Read text from image
            results = self.reader.readtext(preprocessed_path)
            
            # Extract and combine text
            extracted_text = ""
            for (bbox, text, prob) in results:
                extracted_text += text + " "
            
            return extracted_text.strip(), results, preprocessed_path
        
        except Exception as e:
            print(f"Error processing image: {e}")
            return None, None, None
    
    def visualize_detection(self, image_path, results):
        """
        Visualize the detected text regions on the image
        
        Args:
            image_path: Path to the image
            results: OCR detection results
        """
        self.image_processor.visualize_results(image_path, results)


def main():
    # Path to your image with handwritten text
    image_path = "C:\Users\DELL\Documents\SAIRUSHI\Project Research Scientist-1\dataset\111.jpg"  # Replace with your image path
    
    # Initialize the text recognizer
    recognizer = TextRecognizer(languages=['en'])
    
    # Extract text
    extracted_text, results, preprocessed_path = recognizer.recognize_text(image_path)
    
    if extracted_text:
        print("\nExtracted Text:")
        print("-" * 30)
        print(extracted_text)
        print("-" * 30)
        
        # Visualize results
        recognizer.visualize_detection(preprocessed_path, results)
    else:
        print("Failed to extract text from the image.")


if __name__ == "__main__":
    main()