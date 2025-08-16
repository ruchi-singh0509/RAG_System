
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import pytesseract
import easyocr
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class OCREngine:
    """OCR Engine for text extraction from images and scanned documents"""
    
    def __init__(self, tesseract_cmd: str = None, languages: List[str] = None):
        """
        Initialize OCR Engine
        
        Args:
            tesseract_cmd: Path to tesseract executable
            languages: List of languages for OCR (e.g., ['eng', 'fra'])
        """
        self.languages = languages or ['eng']
        self.tesseract_cmd = tesseract_cmd or 'tesseract'
        
        # Configure tesseract
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Initialize EasyOCR reader
        try:
            self.easyocr_reader = easyocr.Reader(self.languages)
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize EasyOCR: {e}")
            self.easyocr_reader = None
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply noise reduction
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to remove noise
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            # Return original image if preprocessing fails
            return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    def extract_text_tesseract(self, image_path: str, preprocess: bool = True) -> Dict[str, Any]:
        """
        Extract text using Tesseract OCR
        
        Args:
            image_path: Path to the image file
            preprocess: Whether to preprocess the image
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            if preprocess:
                image = self.preprocess_image(image_path)
            else:
                image = cv2.imread(image_path)
            
            # Extract text with confidence scores
            data = pytesseract.image_to_data(
                image, 
                output_type=pytesseract.Output.DICT,
                lang='+'.join(self.languages)
            )
            
            # Extract text with bounding boxes
            boxes = pytesseract.image_to_boxes(
                image,
                lang='+'.join(self.languages)
            )
            
            # Get full text
            full_text = pytesseract.image_to_string(
                image,
                lang='+'.join(self.languages)
            )
            
            # Process confidence scores
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': full_text.strip(),
                'confidence': avg_confidence,
                'words': data['text'],
                'confidences': data['conf'],
                'boxes': boxes,
                'method': 'tesseract'
            }
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed for {image_path}: {e}")
            return {
                'text': '',
                'confidence': 0,
                'words': [],
                'confidences': [],
                'boxes': '',
                'method': 'tesseract',
                'error': str(e)
            }
    
    def extract_text_easyocr(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text using EasyOCR
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            if self.easyocr_reader is None:
                raise ValueError("EasyOCR not initialized")
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Extract text
            results = self.easyocr_reader.readtext(image_path)
            
            # Process results
            text_blocks = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                text_blocks.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox
                })
                confidences.append(confidence)
            
            full_text = ' '.join([block['text'] for block in text_blocks])
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': full_text.strip(),
                'confidence': avg_confidence,
                'text_blocks': text_blocks,
                'method': 'easyocr'
            }
            
        except Exception as e:
            logger.error(f"EasyOCR failed for {image_path}: {e}")
            return {
                'text': '',
                'confidence': 0,
                'text_blocks': [],
                'method': 'easyocr',
                'error': str(e)
            }
    
    def extract_text_hybrid(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text using both Tesseract and EasyOCR, choose the best result
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        # Try both OCR engines
        tesseract_result = self.extract_text_tesseract(image_path)
        easyocr_result = self.extract_text_easyocr(image_path)
        
        # Compare results and choose the best one
        tesseract_confidence = tesseract_result.get('confidence', 0)
        easyocr_confidence = easyocr_result.get('confidence', 0)
        
        if tesseract_confidence > easyocr_confidence:
            best_result = tesseract_result
            best_method = 'tesseract'
        else:
            best_result = easyocr_result
            best_method = 'easyocr'
        
        return {
            **best_result,
            'method': f'hybrid_{best_method}',
            'tesseract_confidence': tesseract_confidence,
            'easyocr_confidence': easyocr_confidence,
            'tesseract_text': tesseract_result.get('text', ''),
            'easyocr_text': easyocr_result.get('text', '')
        }
    
    def extract_text_from_image(self, image_path: str, method: str = 'hybrid') -> Dict[str, Any]:
        """
        Extract text from image using specified method
        
        Args:
            image_path: Path to the image file
            method: OCR method ('tesseract', 'easyocr', 'hybrid')
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        logger.info(f"Extracting text from {image_path} using {method}")
        
        if method == 'tesseract':
            result = self.extract_text_tesseract(image_path)
        elif method == 'easyocr':
            result = self.extract_text_easyocr(image_path)
        elif method == 'hybrid':
            result = self.extract_text_hybrid(image_path)
        else:
            raise ValueError(f"Unsupported OCR method: {method}")
        
        # Add metadata
        result['image_path'] = image_path
        result['file_size'] = os.path.getsize(image_path)
        result['image_dimensions'] = self._get_image_dimensions(image_path)
        
        logger.info(f"OCR completed for {image_path}. Confidence: {result.get('confidence', 0):.2f}")
        
        return result
    
    def _get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        """Get image dimensions"""
        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception as e:
            logger.warning(f"Could not get image dimensions for {image_path}: {e}")
            return (0, 0)
    
    def extract_text_from_images(self, image_paths: List[str], method: str = 'hybrid') -> List[Dict[str, Any]]:
        """
        Extract text from multiple images
        
        Args:
            image_paths: List of image file paths
            method: OCR method to use
            
        Returns:
            List of dictionaries containing extracted text and metadata
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.extract_text_from_image(image_path, method)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'text': '',
                    'confidence': 0,
                    'error': str(e)
                })
        
        return results
    
    def get_text_quality_score(self, text: str) -> float:
        """
        Calculate a quality score for extracted text
        
        Args:
            text: Extracted text
            
        Returns:
            Quality score between 0 and 1
        """
        if not text:
            return 0.0
        
        # Calculate various quality metrics
        metrics = {}
        
        # Length score (longer text is generally better)
        metrics['length'] = min(len(text) / 1000, 1.0)
        
        # Word count score
        word_count = len(text.split())
        metrics['word_count'] = min(word_count / 100, 1.0)
        
        # Character diversity score
        unique_chars = len(set(text.lower()))
        total_chars = len(text)
        metrics['diversity'] = unique_chars / max(total_chars, 1)
        
        # Punctuation score
        punctuation_count = sum(1 for c in text if c in '.,!?;:')
        metrics['punctuation'] = min(punctuation_count / 10, 1.0)
        
        # Average word length score
        words = text.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            metrics['avg_word_length'] = min(avg_word_length / 8, 1.0)
        else:
            metrics['avg_word_length'] = 0.0
        
        # Calculate overall score
        overall_score = sum(metrics.values()) / len(metrics)
        
        return overall_score
    
    def validate_ocr_result(self, result: Dict[str, Any]) -> bool:
        """
        Validate OCR result quality
        
        Args:
            result: OCR result dictionary
            
        Returns:
            True if result is valid, False otherwise
        """
        # Check if text was extracted
        text = result.get('text', '')
        if not text:
            return False
        
        # Check confidence threshold
        confidence = result.get('confidence', 0)
        if confidence < 30:  # Low confidence threshold
            return False
        
        # Check text quality score
        quality_score = self.get_text_quality_score(text)
        if quality_score < 0.3:  # Low quality threshold
            return False
        
        return True
