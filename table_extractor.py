
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class TableExtractor:
    """Table Extractor for detecting and extracting tables from documents"""
    
    def __init__(self, tesseract_cmd: str = None):
        """
        Initialize Table Extractor
        
        Args:
            tesseract_cmd: Path to tesseract executable
        """
        self.tesseract_cmd = tesseract_cmd or 'tesseract'
        
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF using multiple methods
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing extracted tables and metadata
        """
        tables = []
        
        try:
            # Method 1: PDFPlumber
            pdfplumber_tables = self._extract_tables_pdfplumber(pdf_path)
            tables.extend(pdfplumber_tables)
            
            # Method 2: PyMuPDF + OCR for image-based tables
            pymupdf_tables = self._extract_tables_pymupdf(pdf_path)
            tables.extend(pymupdf_tables)
            
            logger.info(f"Extracted {len(tables)} tables from {pdf_path}")
            
        except Exception as e:
            logger.error(f"Error extracting tables from PDF {pdf_path}: {e}")
        
        return tables
    
    def _extract_tables_pdfplumber(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables using PDFPlumber"""
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract tables from the page
                    page_tables = page.extract_tables()
                    
                    for table_num, table_data in enumerate(page_tables):
                        if table_data and len(table_data) > 1:  # At least 2 rows
                            # Convert to pandas DataFrame
                            df = pd.DataFrame(table_data[1:], columns=table_data[0])
                            
                            # Clean the DataFrame
                            df = self._clean_dataframe(df)
                            
                            table_info = {
                                'table_id': f"pdfplumber_p{page_num}_t{table_num}",
                                'page_number': page_num + 1,
                                'table_number': table_num + 1,
                                'data': df.to_dict('records'),
                                'dataframe': df,
                                'method': 'pdfplumber',
                                'bbox': page.find_tables()[table_num].bbox if page.find_tables() else None,
                                'text': df.to_string(),
                                'rows': len(df),
                                'columns': len(df.columns)
                            }
                            
                            tables.append(table_info)
                            
        except Exception as e:
            logger.error(f"PDFPlumber extraction failed for {pdf_path}: {e}")
        
        return tables
    
    def _extract_tables_pymupdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables using PyMuPDF and image processing"""
        tables = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get page as image
                mat = fitz.Matrix(2, 2)  # Higher resolution for better detection
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Save temporary image
                temp_img_path = f"./temp/page_{page_num}.png"
                os.makedirs(os.path.dirname(temp_img_path), exist_ok=True)
                
                with open(temp_img_path, "wb") as f:
                    f.write(img_data)
                
                # Extract tables from image
                image_tables = self.extract_tables_from_image(temp_img_path)
                
                for table_num, table_info in enumerate(image_tables):
                    table_info['table_id'] = f"pymupdf_p{page_num}_t{table_num}"
                    table_info['page_number'] = page_num + 1
                    table_info['method'] = 'pymupdf'
                    tables.append(table_info)
                
                # Clean up temporary file
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
            
            doc.close()
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for {pdf_path}: {e}")
        
        return tables
    
    def extract_tables_from_image(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Extract tables from image using computer vision techniques
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of dictionaries containing extracted tables and metadata
        """
        tables = []
        
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Contour-based table detection
            contour_tables = self._detect_tables_by_contours(gray, image_path)
            tables.extend(contour_tables)
            
            # Method 2: Line-based table detection
            line_tables = self._detect_tables_by_lines(gray, image_path)
            tables.extend(line_tables)
            
            # Method 3: OCR-based table detection
            ocr_tables = self._detect_tables_by_ocr(gray, image_path)
            tables.extend(ocr_tables)
            
            logger.info(f"Extracted {len(tables)} tables from image {image_path}")
            
        except Exception as e:
            logger.error(f"Error extracting tables from image {image_path}: {e}")
        
        return tables
    
    def _detect_tables_by_contours(self, gray: np.ndarray, image_path: str) -> List[Dict[str, Any]]:
        """Detect tables using contour detection"""
        tables = []
        
        try:
            # Apply thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size (tables should be reasonably large)
                if w > 100 and h > 50:
                    # Extract region
                    table_region = gray[y:y+h, x:x+w]
                    
                    # Try to extract table data
                    table_data = self._extract_table_data_from_region(table_region)
                    
                    if table_data and len(table_data) > 1:
                        df = pd.DataFrame(table_data)
                        df = self._clean_dataframe(df)
                        
                        table_info = {
                            'table_id': f"contour_{i}",
                            'bbox': (x, y, w, h),
                            'data': df.to_dict('records'),
                            'dataframe': df,
                            'method': 'contour_detection',
                            'text': df.to_string(),
                            'rows': len(df),
                            'columns': len(df.columns)
                        }
                        
                        tables.append(table_info)
        
        except Exception as e:
            logger.error(f"Contour-based table detection failed: {e}")
        
        return tables
    
    def _detect_tables_by_lines(self, gray: np.ndarray, image_path: str) -> List[Dict[str, Any]]:
        """Detect tables using line detection"""
        tables = []
        
        try:
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            
            if lines is not None:
                # Group lines by orientation
                horizontal_lines = []
                vertical_lines = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    
                    if abs(angle) < 10:  # Horizontal lines
                        horizontal_lines.append(line[0])
                    elif abs(angle - 90) < 10:  # Vertical lines
                        vertical_lines.append(line[0])
                
                # Find table regions based on line intersections
                table_regions = self._find_table_regions(horizontal_lines, vertical_lines, gray)
                
                for i, region in enumerate(table_regions):
                    table_data = self._extract_table_data_from_region(region)
                    
                    if table_data and len(table_data) > 1:
                        df = pd.DataFrame(table_data)
                        df = self._clean_dataframe(df)
                        
                        table_info = {
                            'table_id': f"line_{i}",
                            'data': df.to_dict('records'),
                            'dataframe': df,
                            'method': 'line_detection',
                            'text': df.to_string(),
                            'rows': len(df),
                            'columns': len(df.columns)
                        }
                        
                        tables.append(table_info)
        
        except Exception as e:
            logger.error(f"Line-based table detection failed: {e}")
        
        return tables
    
    def _detect_tables_by_ocr(self, gray: np.ndarray, image_path: str) -> List[Dict[str, Any]]:
        """Detect tables using OCR and text layout analysis"""
        tables = []
        
        try:
            # Extract text with bounding boxes
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            
            # Group text by position to identify table-like structures
            text_elements = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:  # Filter by confidence
                    text_elements.append({
                        'text': data['text'][i],
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'w': data['width'][i],
                        'h': data['height'][i],
                        'conf': data['conf'][i]
                    })
            
            # Find table-like structures
            table_regions = self._identify_table_regions(text_elements)
            
            for i, region in enumerate(table_regions):
                table_data = self._extract_table_data_from_region(region)
                
                if table_data and len(table_data) > 1:
                    df = pd.DataFrame(table_data)
                    df = self._clean_dataframe(df)
                    
                    table_info = {
                        'table_id': f"ocr_{i}",
                        'data': df.to_dict('records'),
                        'dataframe': df,
                        'method': 'ocr_detection',
                        'text': df.to_string(),
                        'rows': len(df),
                        'columns': len(df.columns)
                    }
                    
                    tables.append(table_info)
        
        except Exception as e:
            logger.error(f"OCR-based table detection failed: {e}")
        
        return tables
    
    def _extract_table_data_from_region(self, region: np.ndarray) -> Optional[List[List[str]]]:
        """Extract table data from a region using OCR"""
        try:
            # Apply OCR to the region
            text = pytesseract.image_to_string(region)
            
            # Parse text into table structure
            lines = text.strip().split('\n')
            table_data = []
            
            for line in lines:
                if line.strip():
                    # Split by common table separators
                    row = [cell.strip() for cell in line.split('\t')]
                    if len(row) == 1:  # Try other separators
                        row = [cell.strip() for cell in line.split('|')]
                    if len(row) == 1:  # Try spaces
                        row = [cell.strip() for cell in line.split('  ') if cell.strip()]
                    
                    if row:
                        table_data.append(row)
            
            return table_data if table_data else None
            
        except Exception as e:
            logger.error(f"Error extracting table data from region: {e}")
            return None
    
    def _find_table_regions(self, horizontal_lines: List, vertical_lines: List, gray: np.ndarray) -> List[np.ndarray]:
        """Find table regions based on line intersections"""
        regions = []
        
        try:
            # Create a mask for table regions
            mask = np.zeros_like(gray)
            
            # Draw horizontal lines
            for line in horizontal_lines:
                cv2.line(mask, (line[0], line[1]), (line[2], line[3]), 255, 2)
            
            # Draw vertical lines
            for line in vertical_lines:
                cv2.line(mask, (line[0], line[1]), (line[2], line[3]), 255, 2)
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            # Extract regions
            for i in range(1, num_labels):  # Skip background
                x, y, w, h, area = stats[i]
                if area > 1000:  # Minimum area threshold
                    region = gray[y:y+h, x:x+w]
                    regions.append(region)
        
        except Exception as e:
            logger.error(f"Error finding table regions: {e}")
        
        return regions
    
    def _identify_table_regions(self, text_elements: List[Dict]) -> List[np.ndarray]:
        """Identify table regions based on text layout"""
        regions = []
        
        try:
            # Group text elements by y-coordinate (rows)
            y_coords = sorted(set([elem['y'] for elem in text_elements]))
            
            # Find rows with similar y-coordinates
            row_groups = []
            current_group = [y_coords[0]]
            
            for y in y_coords[1:]:
                if abs(y - current_group[-1]) < 20:  # Tolerance for row alignment
                    current_group.append(y)
                else:
                    if len(current_group) > 1:
                        row_groups.append(current_group)
                    current_group = [y]
            
            if len(current_group) > 1:
                row_groups.append(current_group)
            
            # Create table regions from row groups
            for group in row_groups:
                # Find text elements in this row group
                row_elements = [elem for elem in text_elements if elem['y'] in group]
                
                if len(row_elements) > 1:
                    # Sort by x-coordinate
                    row_elements.sort(key=lambda x: x['x'])
                    
                    # Create table data
                    table_data = []
                    for elem in row_elements:
                        table_data.append([elem['text']])
                    
                    regions.append(table_data)
        
        except Exception as e:
            logger.error(f"Error identifying table regions: {e}")
        
        return regions
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize DataFrame"""
        try:
            # Remove empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            # Clean cell values
            for col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace('', np.nan)
            
            # Remove rows with all empty cells
            df = df.dropna(how='all')
            
            # Reset index
            df = df.reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning DataFrame: {e}")
            return df
    
    def validate_table(self, table_info: Dict[str, Any]) -> bool:
        """
        Validate extracted table quality
        
        Args:
            table_info: Table information dictionary
            
        Returns:
            True if table is valid, False otherwise
        """
        try:
            # Check if table has data
            if not table_info.get('data'):
                return False
            
            # Check minimum size
            rows = table_info.get('rows', 0)
            columns = table_info.get('columns', 0)
            
            if rows < 2 or columns < 2:
                return False
            
            # Check if table has meaningful content
            df = table_info.get('dataframe')
            if df is not None:
                # Check for non-empty cells
                non_empty_cells = df.notna().sum().sum()
                total_cells = rows * columns
                
                if non_empty_cells / total_cells < 0.3:  # At least 30% non-empty cells
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating table: {e}")
            return False
    
    def get_table_quality_score(self, table_info: Dict[str, Any]) -> float:
        """
        Calculate quality score for extracted table
        
        Args:
            table_info: Table information dictionary
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            score = 0.0
            
            # Size score
            rows = table_info.get('rows', 0)
            columns = table_info.get('columns', 0)
            size_score = min((rows * columns) / 100, 1.0)
            score += size_score * 0.3
            
            # Structure score
            if table_info.get('dataframe') is not None:
                df = table_info['dataframe']
                # Check for consistent column structure
                col_lengths = [len(str(cell)) for cell in df.iloc[0] if pd.notna(cell)]
                if col_lengths:
                    consistency = 1.0 - (max(col_lengths) - min(col_lengths)) / max(max(col_lengths), 1)
                    score += consistency * 0.3
            
            # Content score
            text = table_info.get('text', '')
            if text:
                content_score = min(len(text) / 1000, 1.0)
                score += content_score * 0.4
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating table quality score: {e}")
            return 0.0
