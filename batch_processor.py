#!/usr/bin/env python3
"""
Batch Classification Script
==========================

Process multiple documents and generate comprehensive reports.
"""

import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from document_classifier import DocumentProcessor

def process_batch(data_dirs: List[str], output_dir: str = "batch_results"):
    """Process all documents in specified directories"""
    
    processor = DocumentProcessor()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    all_results = []
    
    # Collect all PDF files with their expected labels
    pdf_files = []
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if data_path.exists():
            pdfs = list(data_path.glob("*.pdf"))
            # Expected label: 1 if in 'reservs' folder, 0 if in 'no-reservs' folder
            expected_label = 1 if 'reservs' in data_path.name else 0
            pdf_files.extend([(pdf, expected_label, data_path.name) for pdf in pdfs])
    
    print(f"Found {len(pdf_files)} documents to process")
    print(f"Expected reservations: {sum(1 for _, label, _ in pdf_files if label == 1)}")
    print(f"Expected no reservations: {sum(1 for _, label, _ in pdf_files if label == 0)}")
    
    # Process each document
    for i, (pdf_path, expected_label, folder_name) in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        print(f"Expected: {expected_label} ({'Has Reservations' if expected_label == 1 else 'No Reservations'})")
        
        try:
            result = processor.process_document(str(pdf_path))
            result['expected_label'] = expected_label
            result['folder_name'] = folder_name
            result['document_name'] = pdf_path.name
            result['correct_prediction'] = (result['classification'] == expected_label)
            all_results.append(result)
            
            print(f"Predicted: {result['classification']} ({'Has Reservations' if result['classification'] == 1 else 'No Reservations'})")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Correct: {'✓' if result['correct_prediction'] else '✗'}")
            
            # Save individual result
            with open(output_path / f"{pdf_path.stem}_result.json", "w") as f:
                json.dump(result, f, indent=2)
                
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            continue
    
    # Generate comprehensive evaluation report
    generate_evaluation_report(all_results, output_path)
    
    return all_results

def generate_evaluation_report(results: List[Dict], output_dir: Path):
    """Generate comprehensive evaluation report with accuracy metrics"""
    
    if not results:
        print("No results to evaluate!")
        return
    
    # Create summary DataFrame
    summary_data = []
    for result in results:
        summary_data.append({
            'document': result['document_name'],
            'folder': result['folder_name'],
            'expected_label': result['expected_label'],
            'predicted_label': result['classification'],
            'confidence': result['confidence'],
            'samples_used': result['samples_used'],
            'early_stopped': result['early_stopped'],
            'correct': result['correct_prediction']
        })
    
    df = pd.DataFrame(summary_data)
    
    # Calculate comprehensive metrics
    total_docs = len(df)
    correct_predictions = df['correct'].sum()
    accuracy = correct_predictions / total_docs
    
    # Confusion Matrix
    true_positives = len(df[(df['expected_label'] == 1) & (df['predicted_label'] == 1)])
    true_negatives = len(df[(df['expected_label'] == 0) & (df['predicted_label'] == 0)])
    false_positives = len(df[(df['expected_label'] == 0) & (df['predicted_label'] == 1)])
    false_negatives = len(df[(df['expected_label'] == 1) & (df['predicted_label'] == 0)])
    
    # Calculate metrics
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / max(0.001, precision + recall)
    specificity = true_negatives / max(1, true_negatives + false_positives)
    
    # Generate detailed report
    report = f"""
MINERAL RIGHTS CLASSIFICATION EVALUATION REPORT
==============================================

DATASET SUMMARY:
- Total Documents: {total_docs}
- Expected Reservations (Class 1): {df[df['expected_label'] == 1].shape[0]}
- Expected No Reservations (Class 0): {df[df['expected_label'] == 0].shape[0]}

OVERALL ACCURACY: {accuracy:.3f} ({correct_predictions}/{total_docs})

CONFUSION MATRIX:
                    Predicted
                 No Res  Has Res
Actual No Res      {true_negatives:3d}     {false_positives:3d}
Actual Has Res     {false_negatives:3d}     {true_positives:3d}

DETAILED METRICS:
- Accuracy:     {accuracy:.3f}
- Precision:    {precision:.3f} (of predicted reservations, how many were correct)
- Recall:       {recall:.3f} (of actual reservations, how many were found)
- Specificity:  {specificity:.3f} (of actual no-reservations, how many were correctly identified)
- F1 Score:     {f1_score:.3f}

CONFIDENCE STATISTICS:
- Mean Confidence: {df['confidence'].mean():.3f}
- Median Confidence: {df['confidence'].median():.3f}
- Min Confidence: {df['confidence'].min():.3f}
- Max Confidence: {df['confidence'].max():.3f}

SAMPLING STATISTICS:
- Mean Samples Used: {df['samples_used'].mean():.1f}
- Early Stop Rate: {df['early_stopped'].mean():.3f}

PERFORMANCE BY CLASS:
"""
    
    # Add class-specific performance
    for class_label in [0, 1]:
        class_name = "No Reservations" if class_label == 0 else "Has Reservations"
        class_df = df[df['expected_label'] == class_label]
        class_accuracy = class_df['correct'].mean() if len(class_df) > 0 else 0
        class_confidence = class_df['confidence'].mean() if len(class_df) > 0 else 0
        
        report += f"\n{class_name} (Class {class_label}):\n"
        report += f"  - Count: {len(class_df)}\n"
        report += f"  - Accuracy: {class_accuracy:.3f}\n"
        report += f"  - Avg Confidence: {class_confidence:.3f}\n"
    
    # Add misclassification analysis
    report += f"\nMISCLASSIFICATIONS ({total_docs - correct_predictions} total):\n"
    
    # False Positives (predicted reservations but actually no reservations)
    false_pos_df = df[(df['expected_label'] == 0) & (df['predicted_label'] == 1)]
    if len(false_pos_df) > 0:
        report += f"\nFalse Positives ({len(false_pos_df)}) - Predicted reservations but actually none:\n"
        for _, row in false_pos_df.iterrows():
            report += f"  - {row['document']} (conf: {row['confidence']:.3f})\n"
    
    # False Negatives (predicted no reservations but actually has reservations)
    false_neg_df = df[(df['expected_label'] == 1) & (df['predicted_label'] == 0)]
    if len(false_neg_df) > 0:
        report += f"\nFalse Negatives ({len(false_neg_df)}) - Missed reservations:\n"
        for _, row in false_neg_df.iterrows():
            report += f"  - {row['document']} (conf: {row['confidence']:.3f})\n"
    
    # Confidence analysis for correct vs incorrect predictions
    correct_df = df[df['correct'] == True]
    incorrect_df = df[df['correct'] == False]
    
    if len(correct_df) > 0 and len(incorrect_df) > 0:
        report += f"\nCONFIDENCE ANALYSIS:\n"
        report += f"- Correct predictions avg confidence: {correct_df['confidence'].mean():.3f}\n"
        report += f"- Incorrect predictions avg confidence: {incorrect_df['confidence'].mean():.3f}\n"
        report += f"- Confidence difference: {correct_df['confidence'].mean() - incorrect_df['confidence'].mean():.3f}\n"
    
    # Save report
    with open(output_dir / "evaluation_report.txt", "w") as f:
        f.write(report)
    
    # Save detailed CSV
    df.to_csv(output_dir / "detailed_results.csv", index=False)
    
    # Save summary statistics as JSON
    summary_stats = {
        'total_documents': total_docs,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'confusion_matrix': {
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        },
        'confidence_stats': {
            'mean': df['confidence'].mean(),
            'median': df['confidence'].median(),
            'min': df['confidence'].min(),
            'max': df['confidence'].max()
        }
    }
    
    with open(output_dir / "summary_stats.json", "w") as f:
        json.dump(summary_stats, f, indent=2)
    
    print(report)

if __name__ == "__main__":
    # Process all documents and evaluate accuracy
    results = process_batch([
        "data/reservs",
        "data/no-reservs"
    ])
    
    print(f"\nProcessed {len(results)} documents")
    print("Detailed evaluation results saved to batch_results/")
    print("Files created:")
    print("  - evaluation_report.txt (detailed analysis)")
    print("  - detailed_results.csv (per-document results)")
    print("  - summary_stats.json (metrics summary)")
