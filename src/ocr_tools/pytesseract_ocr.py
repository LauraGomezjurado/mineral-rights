import pytesseract
import pdf2image
import time
from typing import Dict, Any
from .base_ocr import BaseOCR

class PytesseractOCR(BaseOCR):
    """Tesseract OCR implementation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Configure tesseract if needed
        if 'tesseract_cmd' in self.config:
            pytesseract.pytesseract.tesseract_cmd = self.config['tesseract_cmd']
    
    def extract_text(self, pdf_path: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path, dpi=300)
            
            all_text = []
            total_confidence = 0
            
            for i, image in enumerate(images):
                # Get text with confidence data
                data = pytesseract.image_to_data(
                    image, 
                    output_type=pytesseract.Output.DICT,
                    config=self.config.get('tesseract_config', '--psm 6')
                )
                
                # Extract text and calculate confidence
                page_text = []
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                
                for j, word in enumerate(data['text']):
                    if word.strip():
                        page_text.append(word)
                
                page_text_str = ' '.join(page_text)
                all_text.append(page_text_str)
                
                if confidences:
                    total_confidence += sum(confidences) / len(confidences)
            
            final_text = '\n\n'.join(all_text)
            avg_confidence = total_confidence / len(images) if images else 0
            
            return {
                'text': final_text,
                'confidence': avg_confidence / 100.0,  # Normalize to 0-1
                'metadata': {
                    'num_pages': len(images),
                    'ocr_engine': 'tesseract',
                    'config': self.config.get('tesseract_config', '--psm 6')
                },
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'text': '',
                'confidence': 0.0,
                'metadata': {'error': str(e), 'ocr_engine': 'tesseract'},
                'processing_time': time.time() - start_time
            } 