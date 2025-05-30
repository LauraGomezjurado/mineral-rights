from paddleocr import PaddleOCR
import fitz  # PyMuPDF
import time
from typing import Dict, Any
from .base_ocr import BaseOCR

class PaddleOCRTool(BaseOCR):
    """PaddleOCR implementation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.ocr = PaddleOCR(
            use_angle_cls=config.get('use_angle_cls', True),
            lang=config.get('lang', 'en'),
            show_log=config.get('show_log', False)
        )
    
    def extract_text(self, pdf_path: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            # Convert PDF to images using PyMuPDF
            doc = fitz.open(pdf_path)
            all_text = []
            total_confidence = 0
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                img_data = pix.tobytes("png")
                
                # Run OCR
                result = self.ocr.ocr(img_data, cls=True)
                
                page_text = []
                page_confidences = []
                
                if result and result[0]:
                    for line in result[0]:
                        if len(line) >= 2:
                            text = line[1][0]
                            confidence = line[1][1]
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
                    'ocr_engine': 'paddleocr',
                    'lang': self.config.get('lang', 'en')
                },
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'text': '',
                'confidence': 0.0,
                'metadata': {'error': str(e), 'ocr_engine': 'paddleocr'},
                'processing_time': time.time() - start_time
            } 