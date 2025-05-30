import easyocr
import fitz
import time
from typing import Dict, Any
from .base_ocr import BaseOCR

class EasyOCRTool(BaseOCR):
    """EasyOCR implementation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.reader = easyocr.Reader(
            config.get('languages', ['en']),
            gpu=config.get('use_gpu', True)
        )
    
    def extract_text(self, pdf_path: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            doc = fitz.open(pdf_path)
            all_text = []
            total_confidence = 0
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                
                # Run OCR
                results = self.reader.readtext(img_data)
                
                page_text = []
                page_confidences = []
                
                for (bbox, text, confidence) in results:
                    page_text.append(text)
                    page_confidences.append(confidence)
                
                all_text.append(' '.join(page_text))
                if page_confidences:
                    total_confidence += sum(page_confidences) / len(page_confidences)
            
            doc.close()
            
            final_text = '\n\n'.join(all_text)
            avg_confidence = total_confidence / len(doc) if len(doc) > 0 else 0
            
            return {
                'text': final_text,
                'confidence': avg_confidence,
                'metadata': {
                    'num_pages': len(doc),
                    'ocr_engine': 'easyocr',
                    'languages': self.config.get('languages', ['en'])
                },
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'text': '',
                'confidence': 0.0,
                'metadata': {'error': str(e), 'ocr_engine': 'easyocr'},
                'processing_time': time.time() - start_time
            } 