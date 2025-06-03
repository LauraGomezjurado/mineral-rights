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
from document_classifier import DocumentProcessor

def process_batch(data_dirs: List[str], output_dir: str = "batch_results"):
    """Process all documents in specified directories"""
    
    processor = DocumentProcessor()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    all_results = []
    
    # Collect all PDF files
    pdf_files = []
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if data_path.exists():
            pdfs = list(data_path.glob("*.pdf"))
            pdf_files.extend([(pdf, data_path.name) for pdf in pdfs])
    
    print(f"Found {len(pdf_files)} documents to process")
    
    # Process each document
    for i, (pdf_path, expected_category) in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        
        try:
            result = processor.process_document(str(pdf_path))
            result['expected_category'] = expected_category
            result['document_name'] = pdf_path.name
            all_results.append(result)
            
            # Save individual result
            with open(output_path / f"{pdf_path.stem}_result.json", "w") as f:
                json.dump(result, f, indent=2)
                
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            continue
    
    # Generate summary report
    generate_summary_report(all_results, output_path)
    
    return all_results

def generate_summary_report(results: List[Dict], output_dir: Path):
    """Generate comprehensive summary report"""
    
    # Create summary DataFrame
    summary_data = []
    for result in results:
        expected_class = 1 if result['expected_category'] == 'reservs' else 0
        summary_data.append({
            'document': result['document_name'],
            'expected_class': expected_class,
            'predicted_class': result['classification'],
            'confidence': result['confidence'],
            'samples_used': result['samples_used'],
            'early_stopped': result['early_stopped'],
            'correct': expected_class == result['classification']
        })
    
    df = pd.DataFrame(summary_data)
    
    # Calculate metrics
    accuracy = df['correct'].mean()
    precision_1 = df[(df['predicted_class'] == 1) & (df['correct'])].shape[0] / max(1, df[df['predicted_class'] == 1].shape[0])
    recall_1 = df[(df['expected_class'] == 1) & (df['correct'])].shape[0] / max(1, df[df['expected_class'] == 1].shape[0])
    f1_score = 2 * (precision_1 * recall_1) / max(0.001, precision_1 + recall_1)
    
    # Generate report
    report = f"""
MINERAL RIGHTS CLASSIFICATION REPORT
====================================

Dataset Summary:
- Total Documents: {len(results)}
- Expected Reservations: {df[df['expected_class'] == 1].shape[0]}
- Expected No Reservations: {df[df['expected_class'] == 0].shape[0]}

Performance Metrics:
- Accuracy: {accuracy:.3f}
- Precision (Reservations): {precision_1:.3f}
- Recall (Reservations): {recall_1:.3f}
- F1 Score: {f1_score:.3f}

Confidence Statistics:
- Mean Confidence: {df['confidence'].mean():.3f}
- Median Confidence: {df['confidence'].median():.3f}
- Min Confidence: {df['confidence'].min():.3f}
- Max Confidence: {df['confidence'].max():.3f}

Sampling Statistics:
- Mean Samples Used: {df['samples_used'].mean():.1f}
- Early Stop Rate: {df['early_stopped'].mean():.3f}

Misclassifications:
"""
    
    # Add misclassification details
    misclassified = df[~df['correct']]
    for _, row in misclassified.iterrows():
        report += f"- {row['document']}: Expected {row['expected_class']}, Got {row['predicted_class']} (conf: {row['confidence']:.3f})\n"
    
    # Save report
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(report)
    
    # Save detailed CSV
    df.to_csv(output_dir / "classification_summary.csv", index=False)
    
    print(report)

if __name__ == "__main__":
    # Process all documents
    results = process_batch([
        "data/reservs",
        "data/no-reservs"
    ])
    
    print(f"\nProcessed {len(results)} documents")
    print("Results saved to batch_results/")
