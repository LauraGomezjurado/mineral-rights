#!/usr/bin/env python3
"""
Premium OCR Evaluation for Mineral Rights Deeds
===============================================

Focused evaluation of state-of-the-art OCR methods for maximum accuracy on legal documents.

- GPT-4o - Latest multimodal model with vision capabilities
- Claude 3.5 Sonnet - Excellent vision + legal understanding
- TrOCR (Printed) - Best transformer-based OCR for typed text
- TrOCR (Handwritten) - Specialized for handwritten annotations
- Donut - OCR-free document understanding
- Surya OCR - New SOTA open-source OCR (if available)

Focus: Maximum accuracy, legal phrase detection, clean markdown output
"""

import os
import sys
import time
import json
import base64
import io
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Data processing
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# PDF and image processing
import fitz  # PyMuPDF
from PIL import Image
try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

# Premium OCR libraries
try:
    import torch
    from transformers import (
        TrOCRProcessor, VisionEncoderDecoderModel,
        DonutProcessor
    )
    TRANSFORMERS_AVAILABLE = True
    print(f" CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f" GPU: {torch.cuda.get_device_name()}")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print(" Transformers not available")

# Surya OCR (new SOTA)
try:
    from surya.ocr import run_ocr
    from surya.model.detection.segformer import load_model as load_det_model, load_processor as load_det_processor
    from surya.model.recognition.model import load_model as load_rec_model
    from surya.model.recognition.processor import load_processor as load_rec_processor
    SURYA_AVAILABLE = True
    print(" Surya OCR available")
except ImportError:
    SURYA_AVAILABLE = False

# LLM APIs
try:
    import openai
    OPENAI_AVAILABLE = True
    print(" OpenAI GPT-4o available")
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
    print(" Claude 3.5 Sonnet available")
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Import utilities
from utils import detect_key_phrases

class PremiumOCREngine:
    """Base class for premium OCR engines"""
    
    def __init__(self, name: str, tier: str = "premium"):
        self.name = name
        self.tier = tier  # "premium", "sota", "llm"
    
    def extract_text(self, image: Image.Image) -> Dict[str, Any]:
        """Extract text from image and return structured result"""
        raise NotImplementedError
    
    def to_legal_markdown(self, text: str) -> str:
        """Convert extracted text to legal-optimized markdown format"""
        if not text.strip():
            return ""
            
        lines = text.split('\n')
        markdown_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Legal document headers
            if any(header in line.upper() for header in [
                'DEED OF', 'WARRANTY DEED', 'QUITCLAIM DEED', 'SPECIAL WARRANTY DEED',
                'MINERAL DEED', 'ROYALTY DEED', 'CONVEYANCE'
            ]):
                markdown_lines.append(f"# {line}")
            
            # Major sections
            elif line.isupper() and len(line) < 80 and len(line) > 3:
                markdown_lines.append(f"## {line}")
            
            # Critical legal phrases
            elif any(phrase in line.lower() for phrase in [
                'know all men', 'to have and to hold', 'in witness whereof',
                'habendum clause', 'granting clause', 'premises considered'
            ]):
                markdown_lines.append(f"**{line}**")
            
            # Mineral rights language (CRITICAL)
            elif any(phrase in line.lower() for phrase in [
                'reserving', 'except', 'subject to', 'mineral rights', 'oil and gas',
                'coal rights', 'mining rights', 'subsurface rights', 'mineral estate',
                'royalty interest', 'working interest', 'lease', 'mineral reservation'
            ]):
                markdown_lines.append(f"***🔍 RESERVATION: {line}***")
            
            # Property descriptions
            elif any(phrase in line.lower() for phrase in [
                'beginning at', 'thence', 'containing', 'acres', 'section', 'township',
                'range', 'quarter', 'lot', 'block', 'subdivision'
            ]):
                markdown_lines.append(f" **{line}**")
            
            # Parties
            elif any(phrase in line.lower() for phrase in [
                'grantor', 'grantee', 'party of the first part', 'party of the second part'
            ]):
                markdown_lines.append(f"👥 **{line}**")
            
            else:
                markdown_lines.append(line)
        
        return '\n\n'.join(markdown_lines)

class TrOCRPremiumEngine(PremiumOCREngine):
    """Microsoft TrOCR - Optimized for legal documents"""
    
    def __init__(self, model_type="printed", use_large=True):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")
            
        size = "large" if use_large else "base"
        model_name = f"microsoft/trocr-{size}-{model_type}"
        super().__init__(f"TrOCR-{size}-{model_type}", "sota")
        
        print(f" Loading {model_name}...")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        # Optimize for GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Enable optimizations
        if torch.cuda.is_available():
            self.model.half()  # Use FP16 for speed
            torch.backends.cudnn.benchmark = True
        
        print(f" {self.name} loaded on {self.device}")
    
    def extract_text(self, image: Image.Image) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            # Process image with optimal settings
            pixel_values = self.processor(
                image, 
                return_tensors="pt"
            ).pixel_values.to(self.device)
            
            # Generate with optimized parameters for legal documents
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=512,  # Longer for legal docs
                    num_beams=8,     # More beams for accuracy
                    early_stopping=True,
                    do_sample=False,  # Deterministic for legal accuracy
                    return_dict_in_generate=True,
                    output_scores=True,
                    repetition_penalty=1.1,
                    length_penalty=1.0
                )
            
            # Decode text
            text = self.processor.batch_decode(
                generated_ids.sequences, 
                skip_special_tokens=True
            )[0]
            
            # Calculate confidence
            if hasattr(generated_ids, 'sequences_scores'):
                confidence = torch.exp(generated_ids.sequences_scores[0]).item()
            else:
                confidence = 0.90  # High default for TrOCR
            
            return {
                'text': text,
                'markdown': self.to_legal_markdown(text),
                'confidence': confidence,
                'processing_time': time.time() - start_time,
                'word_count': len(text.split()),
                'engine': self.name,
                'tier': self.tier
            }
            
        except Exception as e:
            return {
                'text': '',
                'markdown': '',
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'word_count': 0,
                'engine': self.name,
                'tier': self.tier,
                'error': str(e)
            }

class DonutPremiumEngine(PremiumOCREngine):
    """Donut - OCR-free document understanding"""
    
    def __init__(self):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")
            
        super().__init__("Donut-Premium", "sota")
        
        print("🍩 Loading Donut model...")
        self.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        self.model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
        
        # Optimize for GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        if torch.cuda.is_available():
            self.model.half()
            torch.backends.cudnn.benchmark = True
        
        print(f"✅ {self.name} loaded on {self.device}")
    
    def extract_text(self, image: Image.Image) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            # Process image
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            
            # Legal document specific prompt
            task_prompt = "<s_legal_document>"
            decoder_input_ids = self.processor.tokenizer(
                task_prompt, 
                add_special_tokens=False, 
                return_tensors="pt"
            ).input_ids.to(self.device)
            
            # Generate with legal document optimization
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=1024,  # Longer for legal docs
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=5,
                    repetition_penalty=1.2,
                    length_penalty=1.1,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Decode and clean text
            text = self.processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
            
            # Clean Donut output
            import re
            text = re.sub(r'<s_[^>]*>', '', text)
            text = re.sub(r'</s_[^>]*>', '', text)
            text = re.sub(r'<[^>]*>', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            confidence = 0.92  # High confidence for Donut
            
            return {
                'text': text,
                'markdown': self.to_legal_markdown(text),
                'confidence': confidence,
                'processing_time': time.time() - start_time,
                'word_count': len(text.split()),
                'engine': self.name,
                'tier': self.tier
            }
            
        except Exception as e:
            return {
                'text': '',
                'markdown': '',
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'word_count': 0,
                'engine': self.name,
                'tier': self.tier,
                'error': str(e)
            }

class SuryaOCREngine(PremiumOCREngine):
    """Surya OCR - New SOTA open-source OCR"""
    
    def __init__(self):
        if not SURYA_AVAILABLE:
            raise ImportError("Surya OCR not available")
            
        super().__init__("Surya-OCR", "sota")
        
        print(" Loading Surya OCR models...")
        self.det_processor = load_det_processor()
        self.det_model = load_det_model()
        self.rec_model = load_rec_model()
        self.rec_processor = load_rec_processor()
        
        print(f" {self.name} loaded")
    
    def extract_text(self, image: Image.Image) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            # Run Surya OCR
            predictions = run_ocr(
                [image], 
                [None],  # No language specified
                self.det_model,
                self.det_processor,
                self.rec_model,
                self.rec_processor
            )
            
            # Extract text
            if predictions and len(predictions) > 0:
                text_lines = []
                confidences = []
                
                for text_line in predictions[0].text_lines:
                    text_lines.append(text_line.text)
                    confidences.append(text_line.confidence)
                
                text = '\n'.join(text_lines)
                avg_confidence = np.mean(confidences) if confidences else 0.0
            else:
                text = ""
                avg_confidence = 0.0
            
            return {
                'text': text,
                'markdown': self.to_legal_markdown(text),
                'confidence': avg_confidence,
                'processing_time': time.time() - start_time,
                'word_count': len(text.split()),
                'engine': self.name,
                'tier': self.tier
            }
            
        except Exception as e:
            return {
                'text': '',
                'markdown': '',
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'word_count': 0,
                'engine': self.name,
                'tier': self.tier,
                'error': str(e)
            }

class GPT4oPremiumEngine(PremiumOCREngine):
    """GPT-4o - Latest multimodal model with vision capabilities"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        super().__init__(f"GPT-4o-Premium", "llm")
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available")
        
        # Set up OpenAI client
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv('OPENAI_API_KEY')
        )
        
        if not self.client.api_key:
            raise ValueError("No OpenAI API key found")
        
        self.model = model
        print(f"✅ {self.name} initialized with {model}")
    
    def image_to_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def extract_text(self, image: Image.Image) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            base64_image = self.image_to_base64(image)
            
            # Legal-optimized prompt
            prompt = """
            You are an expert legal document OCR system specializing in mineral rights deeds. Extract ALL text with perfect accuracy.

            CRITICAL REQUIREMENTS:
            1. Preserve EXACT spelling, punctuation, and legal terminology
            2. Identify and highlight ALL mineral rights reservation language
            3. Maintain precise document structure and formatting
            4. Include ALL text, even partially visible or faded
            5. Pay special attention to:
               - Granting clauses
               - Habendum clauses  
               - Reservation language ("reserving", "except", "subject to")
               - Property descriptions
               - Mineral rights terminology
               - Oil, gas, coal, mining rights
               - Royalty interests

            FORMAT AS LEGAL MARKDOWN:
            - # for document titles (DEED OF CONVEYANCE, etc.)
            - ## for major sections
            - **bold** for critical legal phrases
            - ***🔍 RESERVATION: text*** for ANY mineral rights language
            - 📍 **text** for property descriptions
            - 👥 **text** for parties (grantor/grantee)

            Extract with legal precision - this affects mineral rights classification.
            """
            
            # Call GPT-4o with legal optimization
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                temperature=0.0,  # Deterministic for legal accuracy
                top_p=1.0
            )
            
            markdown_text = response.choices[0].message.content
            
            # Extract plain text
            import re
            plain_text = re.sub(r'[#*🔍📍👥]', '', markdown_text)
            plain_text = re.sub(r'\s+', ' ', plain_text).strip()
            
            return {
                'text': plain_text,
                'markdown': markdown_text,
                'confidence': 0.98,  # Very high confidence for GPT-4o
                'processing_time': time.time() - start_time,
                'word_count': len(plain_text.split()),
                'engine': self.name,
                'tier': self.tier,
                'tokens_used': response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            return {
                'text': '',
                'markdown': '',
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'word_count': 0,
                'engine': self.name,
                'tier': self.tier,
                'error': str(e)
            }

class ClaudePremiumEngine(PremiumOCREngine):
    """Claude 3.5 Sonnet - Premium legal document analysis"""
    
    def __init__(self, api_key: str = None):
        super().__init__("Claude-3.5-Premium", "llm")
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic not available")
        
        api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("No Anthropic API key found")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        print(f"✅ {self.name} initialized")
    
    def image_to_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def extract_text(self, image: Image.Image) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            base64_image = self.image_to_base64(image)
            
            # Legal-expert prompt
            prompt = """
            You are a specialized legal document OCR expert with deep knowledge of mineral rights law. Extract ALL text with absolute precision.

            LEGAL EXPERTISE REQUIRED:
            1. Perfect accuracy for legal terminology and phrases
            2. Identify ALL mineral rights reservation language with 100% accuracy
            3. Understand legal document structure and hierarchy
            4. Preserve exact legal phrasing and punctuation
            5. Recognize handwritten annotations and signatures
            6. Detect subtle but critical reservation language

            MINERAL RIGHTS FOCUS:
            - Any mention of "reserving", "except", "subject to"
            - Oil, gas, coal, mineral rights language
            - Royalty interests, working interests
            - Subsurface rights, mineral estate
            - Lease references, mineral reservations

            MARKDOWN FORMAT:
            - # for main document titles
            - ## for major legal sections
            - **bold** for critical legal clauses
            - ***🔍 RESERVATION: text*** for ANY mineral rights language
            - 📍 **text** for property descriptions
            - 👥 **text** for parties

            This extraction determines mineral rights classification - be exhaustive and precise.
            """
            
            # Call Claude with legal specialization
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.0,  # Deterministic for legal work
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            markdown_text = response.content[0].text
            
            # Extract plain text
            import re
            plain_text = re.sub(r'[#*🔍📍👥]', '', markdown_text)
            plain_text = re.sub(r'\s+', ' ', plain_text).strip()
            
            return {
                'text': plain_text,
                'markdown': markdown_text,
                'confidence': 0.97,  # Very high confidence for Claude
                'processing_time': time.time() - start_time,
                'word_count': len(plain_text.split()),
                'engine': self.name,
                'tier': self.tier,
                'tokens_used': response.usage.input_tokens + response.usage.output_tokens if response.usage else 0
            }
            
        except Exception as e:
            return {
                'text': '',
                'markdown': '',
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'word_count': 0,
                'engine': self.name,
                'tier': self.tier,
                'error': str(e)
            }

def pdf_to_images(pdf_path: str, dpi: int = 400) -> List[Image.Image]:
    """Convert PDF to high-resolution images optimized for OCR"""
    try:
        if PDF2IMAGE_AVAILABLE:
            # High DPI for premium OCR
            images = pdf2image.convert_from_path(
                pdf_path, 
                dpi=dpi,
                fmt='PNG',
                thread_count=4
            )
            return images
        else:
            raise ImportError("pdf2image not available")
    except Exception as e:
        print(f"pdf2image failed: {e}")
        
        try:
            # Fallback with PyMuPDF
            doc = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Very high resolution for premium OCR
                mat = fitz.Matrix(dpi/72, dpi/72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                images.append(image)
            
            doc.close()
            return images
            
        except Exception as e2:
            print(f"PyMuPDF also failed: {e2}")
            return []

def preprocess_image_premium(image: Image.Image) -> Image.Image:
    """Premium image preprocessing for optimal OCR"""
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Enhance for legal documents
    from PIL import ImageEnhance, ImageFilter
    
    # Slight sharpening for text clarity
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.2)
    
    # Slight contrast enhancement
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)
    
    # Resize for API limits while maintaining quality
    max_size = 2048
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return image

def run_premium_ocr_comparison(pdf_path: str, engines: List[PremiumOCREngine]) -> Dict[str, Any]:
    """Run premium OCR comparison with legal focus"""
    print(f"\n🏆 Premium OCR Processing: {Path(pdf_path).name}")
    
    # Convert PDF with high quality
    print("📄 Converting PDF to high-resolution images...")
    images = pdf_to_images(pdf_path, dpi=400)
    
    if not images:
        print("❌ Failed to convert PDF")
        return {}
    
    print(f"✅ Converted {len(images)} pages at 400 DPI")
    
    # Process first page with premium preprocessing
    image = preprocess_image_premium(images[0])
    
    results = {}
    
    # Run each premium engine
    for engine in engines:
        print(f"🚀 Running {engine.name} ({engine.tier})...")
        
        try:
            result = engine.extract_text(image)
            results[engine.name] = result
            
            # Enhanced progress reporting
            if 'error' in result:
                print(f"❌ {engine.name}: {result['error']}")
            else:
                # Detect mineral rights language
                text = result.get('text', '')
                reservation_indicators = [
                    'reserving', 'except', 'subject to', 'mineral rights',
                    'oil and gas', 'coal rights', 'mining rights'
                ]
                reservations_found = sum(1 for indicator in reservation_indicators if indicator in text.lower())
                
                print(f"✅ {engine.name}: {result['word_count']} words, "
                      f"{result['confidence']:.3f} confidence, "
                      f"{result['processing_time']:.1f}s, "
                      f"{reservations_found} reservation indicators")
                
        except Exception as e:
            print(f"❌ {engine.name} failed: {e}")
            results[engine.name] = {
                'text': '',
                'markdown': '',
                'confidence': 0.0,
                'processing_time': 0.0,
                'word_count': 0,
                'engine': engine.name,
                'tier': engine.tier,
                'error': str(e)
            }
    
    return {
        'pdf_path': pdf_path,
        'pdf_name': Path(pdf_path).name,
        'num_pages': len(images),
        'image_size': image.size,
        'processing_dpi': 400,
        'results': results
    }

def create_premium_dataframe(all_results: List[Dict]) -> pd.DataFrame:
    """Create analysis DataFrame with legal focus"""
    rows = []
    
    for doc_result in all_results:
        doc_name = doc_result['pdf_name']
        
        for engine_name, result in doc_result['results'].items():
            # Enhanced legal phrase detection
            text = result.get('text', '')
            phrases = detect_key_phrases(text)
            
            # Mineral rights specific analysis
            reservation_phrases = [
                'reserving', 'except', 'subject to', 'mineral rights',
                'oil and gas', 'coal rights', 'mining rights', 'subsurface',
                'mineral estate', 'royalty', 'working interest'
            ]
            
            reservation_count = sum(1 for phrase in reservation_phrases if phrase in text.lower())
            has_reservations = reservation_count > 0
            
            row = {
                'document': doc_name,
                'engine': engine_name,
                'tier': result.get('tier', 'unknown'),
                'confidence': result.get('confidence', 0.0),
                'processing_time': result.get('processing_time', 0.0),
                'word_count': result.get('word_count', 0),
                'text_length': len(text),
                'markdown_length': len(result.get('markdown', '')),
                'key_phrases_found': len(phrases),
                'reservation_indicators': reservation_count,
                'likely_has_reservations': has_reservations,
                'has_error': 'error' in result,
                'error_message': result.get('error', ''),
                'tokens_used': result.get('tokens_used', 0)
            }
            
            # Document classification
            if 'reserv' in doc_name.lower() or 'with' in doc_name.lower():
                row['actual_doc_type'] = 'With Reservations'
            else:
                row['actual_doc_type'] = 'Without Reservations'
            
            # Predicted classification based on OCR
            row['predicted_doc_type'] = 'With Reservations' if has_reservations else 'Without Reservations'
            row['classification_correct'] = row['actual_doc_type'] == row['predicted_doc_type']
            
            rows.append(row)
    
    return pd.DataFrame(rows)

def generate_premium_visualizations(df: pd.DataFrame, output_dir: str):
    """Generate premium analysis visualizations"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8')
    
    df_success = df[~df['has_error']]
    
    if df_success.empty:
        print("❌ No successful results to visualize")
        return
    
    # 1. Premium Engine Performance
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Confidence by tier
    sns.boxplot(data=df_success, x='tier', y='confidence', ax=axes[0,0])
    axes[0,0].set_title('Confidence by Engine Tier')
    axes[0,0].set_ylabel('Confidence Score')
    
    # Processing time by engine
    sns.barplot(data=df_success, x='engine', y='processing_time', ax=axes[0,1])
    axes[0,1].set_title('Processing Time by Engine')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].set_ylabel('Time (seconds)')
    
    # Reservation detection accuracy
    sns.barplot(data=df_success, x='engine', y='reservation_indicators', ax=axes[0,2])
    axes[0,2].set_title('Reservation Indicators Found')
    axes[0,2].tick_params(axis='x', rotation=45)
    axes[0,2].set_ylabel('Count')
    
    # Classification accuracy
    accuracy_data = df_success.groupby('engine')['classification_correct'].mean().reset_index()
    sns.barplot(data=accuracy_data, x='engine', y='classification_correct', ax=axes[1,0])
    axes[1,0].set_title('Classification Accuracy')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].set_ylabel('Accuracy')
    axes[1,0].set_ylim(0, 1)
    
    # Word count vs confidence
    sns.scatterplot(data=df_success, x='word_count', y='confidence', hue='tier', ax=axes[1,1])
    axes[1,1].set_title('Word Count vs Confidence')
    axes[1,1].set_xlabel('Words Extracted')
    axes[1,1].set_ylabel('Confidence')
    
    # Cost analysis (for API engines)
    api_engines = df_success[df_success['tier'] == 'llm']
    if not api_engines.empty:
        sns.barplot(data=api_engines, x='engine', y='tokens_used', ax=axes[1,2])
        axes[1,2].set_title('Token Usage (API Engines)')
        axes[1,2].tick_params(axis='x', rotation=45)
        axes[1,2].set_ylabel('Tokens Used')
    else:
        axes[1,2].text(0.5, 0.5, 'No API engines tested', ha='center', va='center', transform=axes[1,2].transAxes)
        axes[1,2].set_title('Token Usage (API Engines)')
    
    plt.tight_layout()
    plt.savefig(output_path / 'premium_ocr_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Legal Classification Matrix
    if 'classification_correct' in df_success.columns:
        plt.figure(figsize=(12, 8))
        
        # Classification accuracy heatmap
        classification_matrix = df_success.pivot_table(
            values='classification_correct',
            index='engine',
            columns='actual_doc_type',
            aggfunc='mean'
        )
        
        sns.heatmap(classification_matrix, annot=True, cmap='RdYlGn', center=0.5, 
                   fmt='.2f', cbar_kws={'label': 'Classification Accuracy'})
        plt.title('Legal Document Classification Accuracy by Engine')
        plt.ylabel('OCR Engine')
        plt.xlabel('Actual Document Type')
        
        plt.tight_layout()
        plt.savefig(output_path / 'classification_accuracy.png', dpi=300, bbox_inches='tight')
        plt.show()

def generate_premium_report(df: pd.DataFrame, all_results: List[Dict], output_dir: str):
    """Generate comprehensive premium analysis report"""
    output_path = Path(output_dir)
    
    df_success = df[~df['has_error']]
    
    report = {
        'evaluation_type': 'Premium OCR for Legal Documents',
        'focus': 'Mineral Rights Classification',
        'summary': {
            'total_documents': len(set(df['document'])),
            'total_engines': len(set(df['engine'])),
            'successful_extractions': len(df_success),
            'failed_extractions': len(df[df['has_error']]),
            'success_rate': len(df_success) / len(df) if len(df) > 0 else 0
        },
        'engine_performance': {},
        'legal_analysis': {},
        'recommendations': [],
        'sample_outputs': {}
    }
    
    # Engine performance by tier
    if not df_success.empty:
        for engine in df_success['engine'].unique():
            engine_data = df_success[df_success['engine'] == engine]
            
            report['engine_performance'][engine] = {
                'tier': engine_data['tier'].iloc[0],
                'avg_confidence': float(engine_data['confidence'].mean()),
                'avg_processing_time': float(engine_data['processing_time'].mean()),
                'avg_word_count': float(engine_data['word_count'].mean()),
                'avg_reservation_indicators': float(engine_data['reservation_indicators'].mean()),
                'classification_accuracy': float(engine_data['classification_correct'].mean()),
                'success_rate': len(engine_data) / len(df[df['engine'] == engine]),
                'avg_tokens_used': float(engine_data['tokens_used'].mean()) if engine_data['tokens_used'].sum() > 0 else 0
            }
        
        # Legal analysis
        report['legal_analysis'] = {
            'documents_with_reservations_detected': int(df_success['likely_has_reservations'].sum()),
            'overall_classification_accuracy': float(df_success['classification_correct'].mean()),
            'best_reservation_detector': df_success.groupby('engine')['reservation_indicators'].mean().idxmax(),
            'most_accurate_classifier': df_success.groupby('engine')['classification_correct'].mean().idxmax()
        }
        
        # Generate recommendations
        best_overall = df_success.groupby('engine').agg({
            'confidence': 'mean',
            'classification_correct': 'mean',
            'processing_time': 'mean'
        }).round(3)
        
        # Weighted score (accuracy * confidence / time)
        best_overall['score'] = (best_overall['confidence'] * best_overall['classification_correct']) / (best_overall['processing_time'] + 0.1)
        best_engine = best_overall['score'].idxmax()
        
        fastest_engine = df_success.groupby('engine')['processing_time'].mean().idxmin()
        most_accurate = df_success.groupby('engine')['classification_correct'].mean().idxmax()
        best_detector = df_success.groupby('engine')['reservation_indicators'].mean().idxmax()
        
        report['recommendations'] = [
            f"🏆 Best Overall Engine: {best_engine} (optimal balance of accuracy, confidence, and speed)",
            f"🎯 Most Accurate Classifier: {most_accurate}",
            f"🔍 Best Reservation Detector: {best_detector}",
            f"⚡ Fastest Processing: {fastest_engine}",
            f"💡 For Production: Use {best_engine} for balanced performance",
            f"🔬 For Research: Use {most_accurate} for maximum accuracy",
            f"⏱️ For Real-time: Use {fastest_engine} for speed-critical applications"
        ]
    
    # Sample outputs with legal focus
    for doc_result in all_results[:2]:
        doc_name = doc_result['pdf_name']
        report['sample_outputs'][doc_name] = {}
        
        for engine_name, result in doc_result['results'].items():
            if not result.get('error'):
                # Focus on reservation language
                markdown = result.get('markdown', '')
                reservation_lines = [line for line in markdown.split('\n') if '🔍 RESERVATION:' in line]
                
                report['sample_outputs'][doc_name][engine_name] = {
                    'confidence': result.get('confidence', 0),
                    'word_count': result.get('word_count', 0),
                    'reservation_indicators': len(reservation_lines),
                    'sample_reservations': reservation_lines[:3],  # First 3 reservation lines
                    'processing_time': result.get('processing_time', 0)
                }
    
    # Save report
    with open(output_path / 'premium_ocr_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print executive summary
    print("\n" + "="*70)
    print("🏆 PREMIUM OCR EVALUATION SUMMARY")
    print("="*70)
    print(f"📄 Documents processed: {report['summary']['total_documents']}")
    print(f"🔧 Premium engines tested: {report['summary']['total_engines']}")
    print(f"✅ Success rate: {report['summary']['success_rate']:.1%}")
    
    if 'legal_analysis' in report:
        print(f"🔍 Classification accuracy: {report['legal_analysis']['overall_classification_accuracy']:.1%}")
        print(f"📊 Reservations detected: {report['legal_analysis']['documents_with_reservations_detected']}")
    
    if report['recommendations']:
        print("\n🎯 EXECUTIVE RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"   {rec}")
    
    print(f"\n💾 Full report: {output_path / 'premium_ocr_report.json'}")

def main():
    parser = argparse.ArgumentParser(description='Premium OCR Evaluation for Mineral Rights Deeds')
    parser.add_argument('--input_dir', type=str, default='data/samples',
                       help='Directory containing PDF files')
    parser.add_argument('--output_dir', type=str, default='data/outputs/premium_ocr_evaluation',
                       help='Output directory')
    parser.add_argument('--engines', nargs='+', 
                       choices=['trocr-printed', 'trocr-handwritten', 'donut', 'surya', 'gpt4o', 'claude'],
                       default=['trocr-printed', 'donut', 'gpt4o', 'claude'],
                       help='Premium OCR engines to test')
    parser.add_argument('--max_docs', type=int, default=5,
                       help='Maximum documents to process')
    
    args = parser.parse_args()
    
    print("🏆 Premium OCR Evaluation for Mineral Rights Deeds")
    print("="*70)
    print("🎯 Focus: Maximum accuracy for legal document classification")
    
    # Initialize premium engines
    engines = []
    
    for engine_name in args.engines:
        try:
            if engine_name == 'trocr-printed':
                engines.append(TrOCRPremiumEngine("printed", use_large=True))
            elif engine_name == 'trocr-handwritten':
                engines.append(TrOCRPremiumEngine("handwritten", use_large=True))
            elif engine_name == 'donut':
                engines.append(DonutPremiumEngine())
            elif engine_name == 'surya':
                engines.append(SuryaOCREngine())
            elif engine_name == 'gpt4o':
                engines.append(GPT4oPremiumEngine())
            elif engine_name == 'claude':
                engines.append(ClaudePremiumEngine())
            
            print(f"✅ {engine_name} initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize {engine_name}: {e}")
    
    if not engines:
        print("❌ No premium engines available")
        return
    
    print(f"\n🚀 Testing {len(engines)} premium engines")
    
    # Find test documents
    input_path = Path(args.input_dir)
    test_docs = []
    
    if input_path.exists():
        test_docs.extend(list(input_path.glob("*.pdf"))[:args.max_docs])
    
    # Add from dataset
    if len(test_docs) < args.max_docs:
        for subdir in ['reservs', 'no-reservs']:
            subdir_path = Path('data') / subdir
            if subdir_path.exists():
                remaining = args.max_docs - len(test_docs)
                test_docs.extend(list(subdir_path.glob("*.pdf"))[:remaining])
                if len(test_docs) >= args.max_docs:
                    break
    
    if not test_docs:
        print("❌ No PDF files found")
        return
    
    print(f"📄 Processing {len(test_docs)} documents with premium OCR")
    
    # Run premium experiments
    all_results = []
    
    for doc_path in test_docs:
        try:
            result = run_premium_ocr_comparison(str(doc_path), engines)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"❌ Failed to process {doc_path}: {e}")
    
    if not all_results:
        print("❌ No successful results")
        return
    
    print(f"\n✅ Completed premium OCR on {len(all_results)} documents")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save results
    with open(output_path / 'premium_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Analysis
    df = create_premium_dataframe(all_results)
    df.to_csv(output_path / 'premium_analysis.csv', index=False)
    
    # Visualizations
    print("📊 Generating premium visualizations...")
    generate_premium_visualizations(df, str(output_path))
    
    # Report
    print("📝 Generating executive report...")
    generate_premium_report(df, all_results, str(output_path))
    
    print(f"\n🎉 Premium OCR evaluation complete!")
    print(f"📁 Results: {output_path}")

if __name__ == "__main__":
    main() 