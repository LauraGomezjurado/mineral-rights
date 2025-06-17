#!/usr/bin/env python3
"""
Test False Positive Fixes
========================

Test the improved classification system on the 9 documents that were false positives
to validate that our fixes improve specificity.
"""

import json
import time
from pathlib import Path
from typing import List, Dict
from document_classifier import DocumentProcessor

def test_false_positive_fixes():
    """Test the improved system on false positive documents"""
    
    print("🧪 TESTING FALSE POSITIVE FIXES")
    print("=" * 60)
    
    # The 9 false positive documents from our analysis
    false_positive_docs = [
        "data/no-reservs/Allegheny DB 92_389.pdf",
        "data/no-reservs/Washington DB 405_547.pdf", 
        "data/no-reservs/Washington DV 677_46.pdf",
        "data/no-reservs/Somerset DB 1820_349.pdf",
        "data/no-reservs/Somerset DB 2163_624.pdf",
        "data/no-reservs/Jefferson DB 397_111.pdf",
        "data/no-reservs/Greene OR 534_885 - 4.23.2025.pdf",
        "data/no-reservs/Potter DB 197_879.pdf",
        "data/no-reservs/Washington DB 384_583.pdf"
    ]
    
    print(f"Testing {len(false_positive_docs)} previously false positive documents...")
    print("Expected result: All should now be classified as 0 (No Reservations)")
    print()
    
    # Initialize processor with improved settings
    processor = DocumentProcessor()
    
    results = []
    fixed_count = 0
    still_wrong_count = 0
    
    for i, doc_path in enumerate(false_positive_docs, 1):
        doc_name = Path(doc_path).name
        print(f"[{i}/{len(false_positive_docs)}] Testing: {doc_name}")
        
        try:
            # Process with improved settings
            result = processor.process_document(
                doc_path,
                max_samples=10,  # Increased from 7
                confidence_threshold=0.7,  # Increased from 0.5 for negative classifications
                page_strategy="sequential_early_stop",
                max_pages=3
            )
            
            predicted_label = result['classification']
            confidence = result['confidence']
            samples_used = result['samples_used']
            
            # Check if we fixed this false positive
            if predicted_label == 0:  # Correct classification (no reservations)
                fixed_count += 1
                print(f"  ✅ FIXED! Now correctly classified as NO reservations")
                print(f"     Confidence: {confidence:.3f}, Samples: {samples_used}")
            else:  # Still wrong
                still_wrong_count += 1
                print(f"  ❌ STILL WRONG - classified as having reservations")
                print(f"     Confidence: {confidence:.3f}, Samples: {samples_used}")
            
            # Store result
            result_data = {
                'document_name': doc_name,
                'original_classification': 1,  # Was false positive
                'new_classification': predicted_label,
                'confidence': confidence,
                'samples_used': samples_used,
                'pages_processed': result['pages_processed'],
                'fixed': predicted_label == 0,
                'early_stopped': result['early_stopped']
            }
            
            results.append(result_data)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'document_name': doc_name,
                'original_classification': 1,
                'new_classification': None,
                'error': str(e),
                'fixed': False
            })
            continue
    
    # Generate summary report
    generate_fix_report(results, fixed_count, still_wrong_count)
    
    return results

def generate_fix_report(results: List[Dict], fixed_count: int, still_wrong_count: int):
    """Generate report on the effectiveness of our fixes"""
    
    total_tested = len([r for r in results if r.get('new_classification') is not None])
    
    print(f"\n📊 FALSE POSITIVE FIX RESULTS")
    print("=" * 50)
    print(f"Documents Tested: {total_tested}")
    print(f"✅ Fixed (now correct): {fixed_count}")
    print(f"❌ Still wrong: {still_wrong_count}")
    print(f"🎯 Fix Rate: {fixed_count/total_tested*100:.1f}%" if total_tested > 0 else "N/A")
    
    if fixed_count > 0:
        print(f"\n🎉 SUCCESS! Fixed {fixed_count} out of {total_tested} false positives")
        
        # Analyze the fixes
        fixed_results = [r for r in results if r.get('fixed', False)]
        if fixed_results:
            avg_samples = sum(r['samples_used'] for r in fixed_results) / len(fixed_results)
            avg_confidence = sum(r['confidence'] for r in fixed_results) / len(fixed_results)
            
            print(f"   Average samples used for fixes: {avg_samples:.1f}")
            print(f"   Average confidence for fixes: {avg_confidence:.3f}")
    
    if still_wrong_count > 0:
        print(f"\n⚠️  Still need to address {still_wrong_count} documents:")
        still_wrong = [r for r in results if not r.get('fixed', False) and r.get('new_classification') is not None]
        for result in still_wrong:
            print(f"   - {result['document_name']} (confidence: {result['confidence']:.3f}, samples: {result['samples_used']})")
    
    # Calculate new specificity estimate
    if total_tested > 0:
        # Original specificity was 18/27 = 0.667
        # If we fixed some false positives, new specificity would be:
        original_true_negatives = 18
        new_true_negatives = original_true_negatives + fixed_count
        total_no_reservation_docs = 27
        
        new_specificity = new_true_negatives / total_no_reservation_docs
        
        print(f"\n📈 ESTIMATED SPECIFICITY IMPROVEMENT:")
        print(f"   Original: 18/27 = 0.667")
        print(f"   New: {new_true_negatives}/27 = {new_specificity:.3f}")
        print(f"   Improvement: +{new_specificity - 0.667:.3f}")
        
        if new_specificity >= 0.90:
            print(f"   🎯 TARGET ACHIEVED! Specificity ≥ 0.90")
        else:
            remaining_fps = total_no_reservation_docs - new_true_negatives
            print(f"   📋 Still need to fix {remaining_fps} more false positives to reach 0.90 target")
    
    # Save detailed results
    output_dir = Path("false_positive_fix_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "fix_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_dir}/fix_test_results.json")

def main():
    """Main test function"""
    
    print("🚀 Starting False Positive Fix Testing...")
    
    try:
        results = test_false_positive_fixes()
        
        fixed_count = len([r for r in results if r.get('fixed', False)])
        total_count = len([r for r in results if r.get('new_classification') is not None])
        
        print(f"\n✅ Testing complete!")
        print(f"   Fixed: {fixed_count}/{total_count} false positives")
        
        if fixed_count == total_count:
            print(f"   🎉 PERFECT! All false positives have been fixed!")
        elif fixed_count > total_count * 0.7:
            print(f"   🎯 GOOD PROGRESS! Most false positives fixed.")
        else:
            print(f"   📋 MORE WORK NEEDED: Consider additional prompt improvements.")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        raise

if __name__ == "__main__":
    main() 