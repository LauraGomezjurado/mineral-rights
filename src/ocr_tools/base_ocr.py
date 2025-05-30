from abc import ABC, abstractmethod
from typing import Dict, Any, List
import json
from pathlib import Path

class BaseOCR(ABC):
    """Base class for all OCR implementations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    def extract_text(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF and return structured result
        
        Returns:
            {
                'text': str,
                'confidence': float,
                'metadata': dict,
                'processing_time': float
            }
        """
        pass
    
    def save_result(self, result: Dict[str, Any], output_path: str):
        """Save OCR result to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False) 