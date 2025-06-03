#!/usr/bin/env python3
"""
Test False Positive Fix
======================

Test the improved prompt on the specific document that was causing false positives.
"""

import json
from pathlib import Path
from document_classifier import DocumentProcessor

def test_false_positive_fix():
    """Test the improved prompt on the problematic document"""
    
    processor = DocumentProcessor()
    
    # The specific document that was causing false positives
    test_file = "data/no-reservs/Allegheny 8277_16 - 4.23.2025.pdf"
    expected_label = 0  # Should be classified as NO reservations
    
    print("🔍 TESTING FALSE POSITIVE FIX")
    print("=" * 50)
    print(f"Document: {test_file}")
    print(f"Expected: {expected_label} (No Reservations)")
    print(f"Problem: Previously classified as 1 due to boilerplate text")
    print("=" * 50)
    
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    try:
        # Process with improved prompt
        result = processor.process_document(
            test_file,
            max_samples=5,
            confidence_threshold=0.6
        )
        
        predicted_label = result['classification']
        confidence = result['confidence']
        is_correct = (predicted_label == expected_label)
        
        print(f"\n📊 RESULTS:")
        print(f"  Predicted: {predicted_label} ({'Has Reservations' if predicted_label == 1 else 'No Reservations'})")
        print(f"  Expected: {expected_label} ({'Has Reservations' if expected_label == 1 else 'No Reservations'})")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Result: {'✅ FIXED!' if is_correct else '❌ STILL WRONG'}")
        
        # Show the reasoning from each sample
        print(f"\n📝 DETAILED REASONING:")
        for i, sample in enumerate(result['detailed_samples'], 1):
            print(f"\nSample {i}:")
            print(f"  Classification: {sample['predicted_class']}")
            print(f"  Individual Confidence: {sample['confidence_score']:.3f}")
            print(f"  Reasoning: {sample['reasoning'][:300]}...")
        
        # Show chunk analysis if available
        if 'chunk_analysis' in result and result['chunk_analysis']:
            print(f"\n📄 CHUNK ANALYSIS:")
            for chunk in result['chunk_analysis']:
                page_num = chunk['page_number']
                chunk_class = chunk['classification']
                chunk_conf = chunk['confidence']
                status = "🎯 FOUND RESERVATIONS!" if chunk_class == 1 else "✓ No reservations"
                print(f"  Page {page_num}: {status} (conf: {chunk_conf:.3f})")
                
                if chunk_class == 1:
                    print(f"    → Analysis stopped here!")
                    break
        
        # Show a snippet of the OCR text to understand what the model saw
        print(f"\n📄 OCR TEXT SNIPPET (first 500 chars):")
        print(f"'{result['ocr_text'][:500]}...'")
        
        # Look for the problematic text
        ocr_text = result['ocr_text'].upper()
        if "RESERVED BY THIS INSTRUMENT" in ocr_text:
            print(f"\n⚠️  FOUND PROBLEMATIC TEXT:")
            # Find the context around the problematic phrase
            start_idx = ocr_text.find("RESERVED BY THIS INSTRUMENT")
            context_start = max(0, start_idx - 100)
            context_end = min(len(ocr_text), start_idx + 200)
            context = result['ocr_text'][context_start:context_end]
            print(f"Context: '...{context}...'")
        
        # Save detailed result for analysis
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "false_positive_test_result.json", "w") as f:
            json.dump({
                'test_file': test_file,
                'expected': expected_label,
                'predicted': predicted_label,
                'correct': is_correct,
                'confidence': confidence,
                'detailed_samples': result['detailed_samples'],
                'ocr_text': result['ocr_text']
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: test_results/false_positive_test_result.json")
        
        return is_correct
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_multiple_edge_cases():
    """Test multiple documents that might cause false positives"""
    
    processor = DocumentProcessor()
    
    # Test files that might have boilerplate language
    test_files = [
        ("data/no-reservs/Allegheny 8277_16 - 4.23.2025.pdf", 0, "Contains boilerplate 'reserved by this instrument'"),
        ("data/no-reservs/Allegheny 1910-768.pdf", 0, "Another no-reservations document"),
        # Add a positive case for comparison
        ("data/reservs/Washington DB 475_646 - 4.23.2025.pdf", 1, "Should have actual reservations"),
    ]
    
    print(f"\n{'='*60}")
    print(f"TESTING MULTIPLE EDGE CASES")
    print(f"{'='*60}")
    
    results = []
    
    for i, (file_path, expected_label, description) in enumerate(test_files, 1):
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"\n[{i}] SKIP: {file_path.name} (file not found)")
            continue
        
        print(f"\n[{i}] Testing: {file_path.name}")
        print(f"Expected: {expected_label} ({'Has Reservations' if expected_label == 1 else 'No Reservations'})")
        print(f"Description: {description}")
        
        try:
            result = processor.process_document(
                str(file_path),
                max_samples=3,  # Fewer samples for faster testing
                confidence_threshold=0.6
            )
            
            predicted_label = result['classification']
            confidence = result['confidence']
            is_correct = (predicted_label == expected_label)
            
            print(f"Predicted: {predicted_label} ({'Has Reservations' if predicted_label == 1 else 'No Reservations'})")
            print(f"Confidence: {confidence:.3f}")
            print(f"Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
            
            results.append({
                'file': file_path.name,
                'expected': expected_label,
                'predicted': predicted_label,
                'correct': is_correct,
                'confidence': confidence,
                'description': description
            })
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Summary
    if results:
        correct_count = sum(1 for r in results if r['correct'])
        total_count = len(results)
        accuracy = correct_count / total_count
        
        print(f"\n{'='*40}")
        print(f"EDGE CASE TEST SUMMARY")
        print(f"{'='*40}")
        print(f"Total tests: {total_count}")
        print(f"Correct: {correct_count}")
        print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        print(f"\nDetailed Results:")
        for r in results:
            status = "✅" if r['correct'] else "❌"
            print(f"  {status} {r['file']}: {r['expected']} → {r['predicted']} (conf: {r['confidence']:.3f})")
            print(f"      {r['description']}")

if __name__ == "__main__":
    # Test the specific problematic document
    success = test_false_positive_fix()
    
    # Test multiple edge cases
    test_multiple_edge_cases()
    
    if success:
        print(f"\n🎉 SUCCESS: The improved prompt appears to have fixed the false positive!")
    else:
        print(f"\n⚠️  The false positive may still exist. Consider further prompt refinements.") 