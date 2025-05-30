#!/usr/bin/env python3
"""
Run OCR comparison experiment on deed PDFs
"""

import argparse
from pathlib import Path
from ocr_benchmark import OCRBenchmark

def main():
    parser = argparse.ArgumentParser(description='Run OCR benchmark on deed PDFs')
    parser.add_argument('--pdf_dir', type=str, required=True, 
                       help='Directory containing PDF files')
    parser.add_argument('--output_dir', type=str, 
                       default='./outputs',
                       help='Output directory for results')
    parser.add_argument('--config', type=str,
                       default='./configs/ocr_config.json',
                       help='Configuration file path')
    parser.add_argument('--max_workers', type=int, default=3,
                       help='Maximum number of worker threads')
    
    args = parser.parse_args()
    
    # Validate inputs
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"Error: PDF directory {pdf_dir} does not exist")
        return
    
    # Run benchmark
    benchmark = OCRBenchmark(args.config)
    benchmark.run_benchmark(
        pdf_directory=str(pdf_dir),
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    main() 