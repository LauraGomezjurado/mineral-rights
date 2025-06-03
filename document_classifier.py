#!/usr/bin/env python3
"""
Mineral Rights Document Classification Agent
===========================================

Self-consistent sampling with confidence scoring for binary classification.
"""

import os
import json
import re
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import anthropic
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import base64

# Set API key
os.environ['ANTHROPIC_API_KEY'] = "your-api-key-here"

@dataclass
class ClassificationSample:
    """Single classification attempt with metadata"""
    predicted_class: int
    reasoning: str
    confidence_score: float
    features: Dict[str, float]
    raw_response: str

@dataclass
class ClassificationResult:
    """Final classification result with metadata"""
    predicted_class: int
    confidence: float
    votes: Dict[int, float]
    samples_used: int
    early_stopped: bool
    all_samples: List[ClassificationSample]

class ConfidenceScorer:
    """Lightweight confidence scoring using logistic regression"""
    
    def __init__(self):
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'sentence_count',
            'trigger_word_presence', 
            'lexical_consistency',
            'format_validity',
            'answer_certainty',
            'past_agreement'
        ]
        
    def extract_features(self, response: str, input_text: str, 
                        past_responses: List[str] = None) -> Dict[str, float]:
        """Extract confidence features from a response"""
        
        # Sentence count
        sentence_count = len(re.findall(r'[.!?]+', response))
        
        # Trigger word presence (uncertainty indicators)
        trigger_words = ['concern', 'issue', 'but', 'however', 'although', 'unclear']
        trigger_presence = sum(1 for word in trigger_words if word.lower() in response.lower())
        
        # Lexical consistency (Jaccard similarity)
        input_words = set(input_text.lower().split())
        response_words = set(response.lower().split())
        if len(input_words.union(response_words)) > 0:
            lexical_consistency = len(input_words.intersection(response_words)) / len(input_words.union(response_words))
        else:
            lexical_consistency = 0.0
            
        # Format validity (structured response)
        format_validity = 1.0 if response.strip().startswith(('Answer:', 'Classification:', 'Result:')) else 0.0
        
        # Answer certainty (hedging terms)
        hedging_terms = ['might', 'probably', 'unclear', 'possibly', 'maybe', 'seems', 'appears']
        answer_certainty = 1.0 - min(1.0, sum(1 for term in hedging_terms if term.lower() in response.lower()) / 3.0)
        
        # Past agreement (similarity to previous high-confidence responses)
        past_agreement = 0.0
        if past_responses:
            similarities = []
            for past_resp in past_responses[-5:]:  # Last 5 responses
                past_words = set(past_resp.lower().split())
                if len(response_words.union(past_words)) > 0:
                    sim = len(response_words.intersection(past_words)) / len(response_words.union(past_words))
                    similarities.append(sim)
            past_agreement = np.mean(similarities) if similarities else 0.0
        
        return {
            'sentence_count': min(sentence_count / 10.0, 1.0),  # Normalize
            'trigger_word_presence': min(trigger_presence / 3.0, 1.0),
            'lexical_consistency': lexical_consistency,
            'format_validity': format_validity,
            'answer_certainty': answer_certainty,
            'past_agreement': past_agreement
        }
    
    def train_initial_model(self):
        """Train with synthetic data for bootstrap"""
        # Generate synthetic training data for initial model
        np.random.seed(42)
        n_samples = 1000
        
        # High confidence samples (good features)
        X_high = np.random.normal([0.7, 0.1, 0.8, 1.0, 0.9, 0.7], 0.1, (n_samples//2, 6))
        y_high = np.ones(n_samples//2)
        
        # Low confidence samples (poor features)  
        X_low = np.random.normal([0.3, 0.8, 0.3, 0.0, 0.2, 0.3], 0.1, (n_samples//2, 6))
        y_low = np.zeros(n_samples//2)
        
        X = np.vstack([X_high, X_low])
        y = np.hstack([y_high, y_low])
        
        # Clip to valid ranges
        X = np.clip(X, 0, 1)
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
    def score_confidence(self, features: Dict[str, float]) -> float:
        """Score confidence for a response"""
        if not self.is_trained:
            self.train_initial_model()
            
        feature_vector = np.array([[features[name] for name in self.feature_names]])
        feature_vector_scaled = self.scaler.transform(feature_vector)
        confidence = self.model.predict_proba(feature_vector_scaled)[0][1]
        return float(confidence)

class MineralRightsClassifier:
    """Main classification agent with self-consistent sampling"""
    
    def __init__(self, api_key: str = None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.confidence_scorer = ConfidenceScorer()
        self.past_high_confidence_responses = []
        
    def create_classification_prompt(self, ocr_text: str) -> str:
        """Create the prompt template for classification"""
        
        prompt = f"""You are a legal document analyst specializing in mineral rights. Your task is to classify whether a deed document contains mineral rights reservations.

DOCUMENT TEXT (from OCR):
{ocr_text}

CLASSIFICATION TASK:
Analyze this document and determine if it contains any mineral rights reservations. Look for language that:
- Reserves, excepts, or retains mineral rights
- Mentions coal, oil, gas, or other mineral reservations  
- References previous deeds that reserved minerals
- Contains "subject to" clauses regarding minerals

RESPONSE FORMAT:
Answer: [0 or 1]
Reasoning: [Provide detailed explanation of your classification decision, citing specific text from the document that supports your conclusion]

Where:
- 0 = No mineral rights reservations found
- 1 = Mineral rights reservations are present

Be thorough in your analysis and cite specific phrases or clauses that support your decision."""

        return prompt
    
    def extract_classification(self, response: str) -> Tuple[Optional[int], str]:
        """Extract classification and reasoning from model response"""
        
        # Look for Answer: pattern
        answer_match = re.search(r'Answer:\s*([01])', response, re.IGNORECASE)
        if answer_match:
            classification = int(answer_match.group(1))
        else:
            # Fallback: look for standalone 0 or 1
            number_matches = re.findall(r'\b([01])\b', response)
            if number_matches:
                classification = int(number_matches[0])
            else:
                return None, response
        
        # Extract reasoning
        reasoning_match = re.search(r'Reasoning:\s*(.*?)(?:\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            reasoning = response
            
        return classification, reasoning
    
    def generate_sample(self, ocr_text: str, temperature: float = 0.7) -> Optional[ClassificationSample]:
        """Generate a single classification sample"""
        
        prompt = self.create_classification_prompt(ocr_text)
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=temperature,
                messages=[{
                    "role": "user", 
                    "content": prompt
                }]
            )
            
            raw_response = response.content[0].text
            predicted_class, reasoning = self.extract_classification(raw_response)
            
            if predicted_class is None:
                return None
                
            # Extract features for confidence scoring
            features = self.confidence_scorer.extract_features(
                raw_response, 
                ocr_text, 
                self.past_high_confidence_responses
            )
            
            # Score confidence
            confidence_score = self.confidence_scorer.score_confidence(features)
            
            return ClassificationSample(
                predicted_class=predicted_class,
                reasoning=reasoning,
                confidence_score=confidence_score,
                features=features,
                raw_response=raw_response
            )
            
        except Exception as e:
            print(f"Error generating sample: {e}")
            return None
    
    def classify_document(self, ocr_text: str, max_samples: int = 10, 
                         confidence_threshold: float = 0.7) -> ClassificationResult:
        """Classify document using self-consistent sampling"""
        
        votes = {0: 0.0, 1: 0.0}
        all_samples = []
        early_stopped = False
        
        for i in range(max_samples):
            print(f"Generating sample {i+1}/{max_samples}...")
            
            # Generate sample with some temperature variation
            temperature = 0.5 + (i * 0.1)  # Increase diversity over time
            sample = self.generate_sample(ocr_text, temperature)
            
            if sample is None:
                continue
                
            all_samples.append(sample)
            
            # Add weighted vote
            votes[sample.predicted_class] += sample.confidence_score
            
            # Store high-confidence responses for future reference
            if sample.confidence_score > 0.8:
                self.past_high_confidence_responses.append(sample.raw_response)
                # Keep only recent high-confidence responses
                if len(self.past_high_confidence_responses) > 20:
                    self.past_high_confidence_responses.pop(0)
            
            # Check early stopping condition
            total_votes = sum(votes.values())
            if total_votes > 0:
                leading_class = max(votes.keys(), key=lambda k: votes[k])
                leading_proportion = votes[leading_class] / total_votes
                
                if leading_proportion >= confidence_threshold and i >= 2:  # Minimum 3 samples
                    early_stopped = True
                    break
        
        # Determine final classification
        if sum(votes.values()) > 0:
            predicted_class = max(votes.keys(), key=lambda k: votes[k])
            final_confidence = votes[predicted_class] / sum(votes.values())
        else:
            predicted_class = 0  # Default to no reservations
            final_confidence = 0.0
            
        return ClassificationResult(
            predicted_class=predicted_class,
            confidence=final_confidence,
            votes=votes,
            samples_used=len(all_samples),
            early_stopped=early_stopped,
            all_samples=all_samples
        )

class DocumentProcessor:
    """Complete pipeline from PDF to classification"""
    
    def __init__(self, api_key: str = None):
        self.classifier = MineralRightsClassifier(api_key)
        
    def pdf_to_image(self, pdf_path: str) -> Image.Image:
        """Convert PDF to high-quality image"""
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        mat = fitz.Matrix(2, 2)  # 2x zoom for quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        doc.close()
        return Image.open(BytesIO(img_data))
    
    def extract_text_with_claude(self, image: Image.Image) -> str:
        """Extract text using Claude OCR"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        response = self.classifier.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": "Extract ALL text from this legal deed document. Pay special attention to any mineral rights reservations. Format as clean text."
                    }
                ]
            }]
        )
        
        return response.content[0].text
    
    def process_document(self, pdf_path: str, max_samples: int = 10, 
                        confidence_threshold: float = 0.7) -> Dict:
        """Complete pipeline: PDF -> OCR -> Classification"""
        
        print(f"Processing: {pdf_path}")
        
        # Step 1: Convert PDF to image
        image = self.pdf_to_image(pdf_path)
        
        # Step 2: Extract text with Claude OCR
        print("Extracting text with Claude OCR...")
        ocr_text = self.extract_text_with_claude(image)
        
        # Step 3: Classify with self-consistent sampling
        print("Classifying document...")
        classification_result = self.classifier.classify_document(
            ocr_text, max_samples, confidence_threshold
        )
        
        return {
            'document_path': pdf_path,
            'ocr_text': ocr_text,
            'classification': classification_result.predicted_class,
            'confidence': classification_result.confidence,
            'votes': classification_result.votes,
            'samples_used': classification_result.samples_used,
            'early_stopped': classification_result.early_stopped,
            'detailed_samples': [
                {
                    'predicted_class': s.predicted_class,
                    'reasoning': s.reasoning,
                    'confidence_score': s.confidence_score,
                    'features': s.features
                }
                for s in classification_result.all_samples
            ]
        }

def main():
    """Example usage"""
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Process single document
    pdf_path = "data/reservs/Washington DB 405_547.pdf"
    result = processor.process_document(pdf_path)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION RESULT")
    print(f"{'='*60}")
    print(f"Document: {result['document_path']}")
    print(f"Classification: {result['classification']} ({'Has Reservations' if result['classification'] == 1 else 'No Reservations'})")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Samples Used: {result['samples_used']}")
    print(f"Early Stopped: {result['early_stopped']}")
    print(f"Vote Distribution: {result['votes']}")
    
    # Save detailed results
    output_dir = Path("classification_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / f"result_{Path(pdf_path).stem}.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_dir}")

if __name__ == "__main__":
    main()
