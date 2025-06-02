#!/usr/bin/env python3
"""
Simulate the complete mineral rights classification pipeline
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from utils import load_prompts, extract_sample_text, detect_key_phrases

def simulate_ocr_extraction(pdf_path: str) -> Dict[str, Any]:
    """Simulate OCR extraction with multiple tools"""
    print("🔍 Step 1: OCR Text Extraction")
    print("   • Running TrOCR (printed)...")
    print("   • Running TrOCR (handwritten)...")
    print("   • Running Donut...")
    print("   • Running Tesseract...")
    
    # Simulate processing time
    time.sleep(2)
    
    # Extract actual sample text for demonstration
    sample_text = extract_sample_text(pdf_path)
    
    ocr_results = {
        "trocr_printed": {
            "text": sample_text,
            "confidence": 0.92,
            "processing_time": 3.2
        },
        "trocr_handwritten": {
            "text": sample_text,
            "confidence": 0.87,
            "processing_time": 3.8
        },
        "donut": {
            "text": sample_text,
            "confidence": 0.89,
            "processing_time": 2.1
        },
        "tesseract": {
            "text": sample_text,
            "confidence": 0.76,
            "processing_time": 1.5
        }
    }
    
    # Select best OCR result
    best_ocr = max(ocr_results.items(), key=lambda x: x[1]["confidence"])
    
    print(f"   ✅ Best OCR: {best_ocr[0]} (confidence: {best_ocr[1]['confidence']:.2f})")
    
    return {
        "all_results": ocr_results,
        "best_result": {
            "tool": best_ocr[0],
            "text": best_ocr[1]["text"],
            "confidence": best_ocr[1]["confidence"]
        }
    }

def simulate_text_normalization(text: str) -> Dict[str, Any]:
    """Simulate text cleaning and normalization"""
    print("\n🧹 Step 2: Text Normalization")
    print("   • Removing OCR artifacts...")
    print("   • Standardizing legal terminology...")
    print("   • Fixing common OCR errors...")
    
    time.sleep(1)
    
    # Simulate normalization improvements
    normalized_text = text.replace("  ", " ").strip()
    
    improvements = [
        "Fixed 3 spacing issues",
        "Corrected 2 legal term spellings",
        "Removed 1 OCR artifact"
    ]
    
    print(f"   ✅ Normalization complete: {', '.join(improvements)}")
    
    return {
        "original_text": text,
        "normalized_text": normalized_text,
        "improvements": improvements
    }

def simulate_phrase_detection(text: str) -> Dict[str, Any]:
    """Simulate key phrase detection"""
    print("\n🔍 Step 3: Key Phrase Detection")
    print("   • Scanning for reservation phrases...")
    print("   • Analyzing legal language patterns...")
    
    time.sleep(1)
    
    # Detect actual phrases in the text
    detected_phrases = detect_key_phrases(text)
    
    print(f"   ✅ Found {len(detected_phrases)} key phrases")
    for phrase in detected_phrases:
        print(f"      • '{phrase['phrase']}' (confidence: {phrase['confidence']:.2f})")
    
    return {
        "detected_phrases": detected_phrases,
        "has_reservations": len(detected_phrases) > 0
    }

def simulate_classification(text: str, phrases: List[Dict]) -> Dict[str, Any]:
    """Simulate LLM-based classification"""
    print("\n🤖 Step 4: LLM Classification")
    print("   • Loading classification prompt...")
    print("   • Analyzing document context...")
    print("   • Generating classification reasoning...")
    
    time.sleep(2)
    
    # Load classification prompt
    prompts = load_prompts()
    
    # Simulate classification logic
    has_reservation_phrases = len(phrases) > 0
    
    if has_reservation_phrases:
        classification = "WITH_RESERVATIONS"
        confidence = 0.91
        reasoning = f"Document contains {len(phrases)} clear reservation phrases indicating mineral rights are reserved by the grantor."
    else:
        classification = "WITHOUT_RESERVATIONS"
        confidence = 0.88
        reasoning = "No explicit mineral rights reservation language found in the document."
    
    print(f"   ✅ Classification: {classification} (confidence: {confidence:.2f})")
    
    return {
        "classification": classification,
        "confidence": confidence,
        "reasoning": reasoning,
        "prompt_used": prompts["classify"][:100] + "..."
    }

def simulate_verification(classification_result: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Simulate self-verification step"""
    print("\n✅ Step 5: Self-Verification")
    print("   • Double-checking classification...")
    print("   • Analyzing potential edge cases...")
    print("   • Calculating final confidence...")
    
    time.sleep(1)
    
    # Load verification prompt
    prompts = load_prompts()
    
    # Simulate verification logic
    original_confidence = classification_result["confidence"]
    
    # Simulate verification adjustments
    if original_confidence > 0.9:
        final_confidence = original_confidence
        verification_notes = "High confidence classification confirmed"
    else:
        final_confidence = max(0.7, original_confidence - 0.05)
        verification_notes = "Moderate confidence - recommend human review"
    
    print(f"   ✅ Verification complete: {final_confidence:.2f} confidence")
    
    return {
        "original_confidence": original_confidence,
        "final_confidence": final_confidence,
        "verification_notes": verification_notes,
        "verified_classification": classification_result["classification"]
    }

def run_pipeline_simulation(pdf_path: str, output_path: str = None):
    """Run complete pipeline simulation"""
    print("🚀 Starting Mineral Rights Classification Pipeline")
    print(f"📄 Processing: {Path(pdf_path).name}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: OCR Extraction
    ocr_result = simulate_ocr_extraction(pdf_path)
    
    # Step 2: Text Normalization
    norm_result = simulate_text_normalization(ocr_result["best_result"]["text"])
    
    # Step 3: Phrase Detection
    phrase_result = simulate_phrase_detection(norm_result["normalized_text"])
    
    # Step 4: Classification
    class_result = simulate_classification(
        norm_result["normalized_text"], 
        phrase_result["detected_phrases"]
    )
    
    # Step 5: Verification
    verify_result = simulate_verification(class_result, norm_result["normalized_text"])
    
    # Compile final results
    total_time = time.time() - start_time
    
    final_result = {
        "input_file": str(pdf_path),
        "processing_time": total_time,
        "pipeline_steps": {
            "ocr_extraction": ocr_result,
            "text_normalization": norm_result,
            "phrase_detection": phrase_result,
            "classification": class_result,
            "verification": verify_result
        },
        "final_output": {
            "classification": verify_result["verified_classification"],
            "confidence": verify_result["final_confidence"],
            "key_phrases_found": len(phrase_result["detected_phrases"]),
            "reasoning": class_result["reasoning"]
        }
    }
    
    print("\n" + "=" * 60)
    print("🎯 FINAL CLASSIFICATION RESULT")
    print("=" * 60)
    print(f"📋 Document: {Path(pdf_path).name}")
    print(f"🏷️  Classification: {final_result['final_output']['classification']}")
    print(f"📊 Confidence: {final_result['final_output']['confidence']:.2f}")
    print(f"🔍 Key Phrases: {final_result['final_output']['key_phrases_found']}")
    print(f"💭 Reasoning: {final_result['final_output']['reasoning']}")
    print(f"⏱️  Total Time: {total_time:.1f}s")
    
    # Save results if output path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_file}")
    
    return final_result

def main():
    parser = argparse.ArgumentParser(description='Simulate mineral rights classification pipeline')
    parser.add_argument('--input', type=str, required=True,
                       help='Input PDF file path')
    parser.add_argument('--output', type=str,
                       help='Output JSON file path (optional)')
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file {input_path} does not exist")
        return
    
    # Set default output path
    output_path = args.output
    if not output_path:
        output_path = f"data/outputs/{input_path.stem}_classification.json"
    
    # Run simulation
    try:
        run_pipeline_simulation(str(input_path), output_path)
    except Exception as e:
        print(f"❌ Pipeline simulation failed: {e}")

if __name__ == "__main__":
    main() 