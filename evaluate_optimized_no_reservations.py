#!/usr/bin/env python3
"""
Optimized No-Reservations Detection Evaluation
==============================================

Evaluation focused on maximizing detection of documents WITHOUT reservations
while accepting some false negatives on documents WITH reservations.
"""

import json
import time
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from document_classifier import DocumentProcessor

def evaluate_optimized_approach(data_dirs: List[str], output_dir: str = "optimized_evaluation_results"):
    """Evaluate the optimized pipeline focused on no-reservation detection"""
    
    processor = DocumentProcessor()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("🔍 OPTIMIZED NO-RESERVATIONS DETECTION EVALUATION")
    print("=" * 70)
    print("Strategy: Conservative bias toward detecting documents WITHOUT reservations")
    print("Goal: Minimize false positives (docs wrongly classified as having reservations)")
    print("Acceptable: Some false negatives (missing actual reservations)")
    print("=" * 70)
    
    # Collect all PDF files with their expected labels
    pdf_files = []
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if data_path.exists():
            pdfs = list(data_path.glob("*.pdf"))
            folder_name = data_path.name
            if folder_name == 'reservs':
                expected_label = 1
            elif folder_name == 'no-reservs':
                expected_label = 0
            else:
                print(f"Warning: Unknown folder '{folder_name}', skipping...")
                continue
            pdf_files.extend([(pdf, expected_label, folder_name) for pdf in pdfs])
    
    print(f"📁 DATASET OVERVIEW:")
    print(f"  Total documents: {len(pdf_files)}")
    print(f"  Expected reservations: {sum(1 for _, label, _ in pdf_files if label == 1)}")
    print(f"  Expected no reservations: {sum(1 for _, label, _ in pdf_files if label == 0)}")
    print()
    
    # Process each document with optimized settings
    all_results = []
    start_time = time.time()
    
    # Track performance by category
    no_reservs_correct = 0
    no_reservs_total = 0
    reservs_correct = 0
    reservs_total = 0
    
    for i, (pdf_path, expected_label, folder_name) in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        print(f"Expected: {expected_label} ({'Has Reservations' if expected_label == 1 else 'No Reservations'})")
        
        try:
            # Process with optimized settings for no-reservation detection
            result = processor.process_document(
                str(pdf_path),
                max_samples=7,  # More samples for better confidence
                confidence_threshold=0.5,  # Lower threshold to be more conservative
                page_strategy="sequential_early_stop",
                max_pages=3  # Focus on first few pages where reservations usually appear
            )
            
            predicted_label = result['classification']
            confidence = result['confidence']
            is_correct = (predicted_label == expected_label)
            
            # Track category-specific performance
            if expected_label == 0:  # No reservations expected
                no_reservs_total += 1
                if is_correct:
                    no_reservs_correct += 1
                    print(f"  ✅ CORRECTLY identified as NO reservations")
                else:
                    print(f"  ❌ FALSE POSITIVE - wrongly identified as having reservations")
            else:  # Reservations expected
                reservs_total += 1
                if is_correct:
                    reservs_correct += 1
                    print(f"  ✅ CORRECTLY identified reservations")
                else:
                    print(f"  ⚠️  FALSE NEGATIVE - missed actual reservations (acceptable)")
            
            print(f"  Predicted: {predicted_label} ({'Has Reservations' if predicted_label == 1 else 'No Reservations'})")
            print(f"  Confidence: {confidence:.3f}")
            
            # Store result
            result_data = {
                'document_name': pdf_path.name,
                'folder_name': folder_name,
                'expected_label': expected_label,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'correct_prediction': is_correct,
                'pages_processed': result['pages_processed'],
                'samples_used': result['samples_used'],
                'processing_time': None
            }
            
            all_results.append(result_data)
            
            # Show running performance on no-reservations (our main goal)
            if no_reservs_total > 0:
                no_reservs_accuracy = no_reservs_correct / no_reservs_total
                print(f"  📊 No-Reservations Accuracy: {no_reservs_accuracy:.3f} ({no_reservs_correct}/{no_reservs_total})")
            
        except Exception as e:
            print(f"   ERROR: {e}")
            all_results.append({
                'document_name': pdf_path.name,
                'folder_name': folder_name,
                'expected_label': expected_label,
                'predicted_label': None,
                'confidence': 0.0,
                'correct_prediction': False,
                'error': str(e)
            })
            continue
    
    total_time = time.time() - start_time
    
    # Generate focused evaluation report
    generate_optimized_report(all_results, output_path, total_time, 
                            no_reservs_correct, no_reservs_total,
                            reservs_correct, reservs_total)
    
    return all_results

def generate_optimized_report(results: List[Dict], output_dir: Path, total_time: float,
                            no_reservs_correct: int, no_reservs_total: int,
                            reservs_correct: int, reservs_total: int):
    """Generate report focused on no-reservations detection performance"""
    
    # Filter out error results
    valid_results = [r for r in results if r['predicted_label'] is not None]
    
    if not valid_results:
        print("❌ No valid results to evaluate!")
        return
    
    df = pd.DataFrame(valid_results)
    
    # Calculate key metrics
    total_docs = len(valid_results)
    overall_accuracy = df['correct_prediction'].sum() / total_docs
    
    # Focus on no-reservations performance (main goal)
    no_reservs_accuracy = no_reservs_correct / no_reservs_total if no_reservs_total > 0 else 0
    reservs_accuracy = reservs_correct / reservs_total if reservs_total > 0 else 0
    
    # Confusion matrix
    true_positives = len(df[(df['expected_label'] == 1) & (df['predicted_label'] == 1)])
    true_negatives = len(df[(df['expected_label'] == 0) & (df['predicted_label'] == 0)])
    false_positives = len(df[(df['expected_label'] == 0) & (df['predicted_label'] == 1)])
    false_negatives = len(df[(df['expected_label'] == 1) & (df['predicted_label'] == 0)])
    
    # Our main metric: specificity (ability to correctly identify no-reservations)
    specificity = true_negatives / max(1, true_negatives + false_positives)
    # Secondary metric: sensitivity (ability to detect reservations)
    sensitivity = true_positives / max(1, true_positives + false_negatives)
    
    # Generate report
    report = f"""
🔍 OPTIMIZED NO-RESERVATIONS DETECTION EVALUATION REPORT
========================================================

🎯 PRIMARY GOAL: Maximize detection of documents WITHOUT reservations
📊 SECONDARY GOAL: Acceptable performance on documents WITH reservations

PERFORMANCE SUMMARY:
==================
📈 NO-RESERVATIONS ACCURACY: {no_reservs_accuracy:.3f} ({no_reservs_correct}/{no_reservs_total})
📈 RESERVATIONS ACCURACY: {reservs_accuracy:.3f} ({reservs_correct}/{reservs_total})
📈 OVERALL ACCURACY: {overall_accuracy:.3f} ({df['correct_prediction'].sum()}/{total_docs})

KEY METRICS:
===========
🎯 SPECIFICITY (No-Reservations Detection): {specificity:.3f}
   - This is our PRIMARY metric - how well we identify docs WITHOUT reservations
   - Target: > 0.90 (vs. current {specificity:.3f})

⚖️  SENSITIVITY (Reservations Detection): {sensitivity:.3f}
   - This is SECONDARY - some false negatives are acceptable
   - Current: {sensitivity:.3f}

CONFUSION MATRIX:
================
                    Predicted
                 No Res  Has Res
Actual No Res      {true_negatives:2d}      {false_positives:2d}
Actual Has Res     {false_negatives:2d}      {true_positives:2d}

IMPROVEMENT ANALYSIS:
====================
🔴 False Positives (Main Problem): {false_positives}
   - Documents WITHOUT reservations wrongly classified as having them
   - These are the critical errors we want to eliminate

🟡 False Negatives (Acceptable): {false_negatives}
   - Documents WITH reservations that we missed
   - Less critical for your use case

PROCESSING EFFICIENCY:
=====================
⏱️  Total Time: {total_time:.1f} seconds
⏱️  Average per Document: {total_time/total_docs:.1f} seconds
📄 Average Pages Processed: {df['pages_processed'].mean():.1f}
🔄 Average Samples Used: {df['samples_used'].mean():.1f}

RECOMMENDED NEXT STEPS:
======================
"""

    if false_positives > 0:
        report += f"""
🔧 PRIORITY: Reduce {false_positives} false positives
   1. Analyze the {false_positives} documents wrongly classified as having reservations
   2. Identify common patterns in boilerplate language being misclassified
   3. Enhance prompt with specific examples from these false positives
   4. Consider raising confidence threshold for positive classifications
"""
    else:
        report += """
✅ EXCELLENT: Zero false positives achieved!
   - Consider slight threshold adjustments to catch more true reservations
   - Monitor performance on larger dataset
"""

    if specificity < 0.90:
        report += f"""
📈 SPECIFICITY IMPROVEMENT NEEDED:
   - Current: {specificity:.3f}, Target: > 0.90
   - Focus on conservative classification strategies
   - Enhance boilerplate detection patterns
"""
    else:
        report += f"""
🎉 SPECIFICITY TARGET ACHIEVED: {specificity:.3f} > 0.90
   - Excellent performance on main objective
"""

    print(report)
    
    # Save detailed report
    with open(output_dir / "optimized_evaluation_report.txt", "w") as f:
        f.write(report)
    
    # Save detailed results
    df.to_csv(output_dir / "optimized_detailed_results.csv", index=False)
    
    # Analyze false positives in detail
    if false_positives > 0:
        false_positive_docs = df[(df['expected_label'] == 0) & (df['predicted_label'] == 1)]
        print(f"\n🔍 ANALYZING {false_positives} FALSE POSITIVES:")
        for _, row in false_positive_docs.iterrows():
            print(f"  - {row['document_name']} (confidence: {row['confidence']:.3f})")
        
        false_positive_docs.to_csv(output_dir / "false_positives_analysis.csv", index=False)

def main():
    """Main evaluation function"""
    
    # Define data directories
    data_dirs = [
        "data/no-reservs",  # Documents WITHOUT reservations (our main focus)
        "data/reservs"      # Documents WITH reservations (secondary)
    ]
    
    print("🚀 Starting Optimized No-Reservations Detection Evaluation...")
    
    # Run evaluation
    results = evaluate_optimized_approach(data_dirs)
    
    print("\n✅ Evaluation complete! Check 'optimized_evaluation_results/' for detailed analysis.")

if __name__ == "__main__":
    main() 