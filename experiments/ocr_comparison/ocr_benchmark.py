import json
import time
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ocr_tools.pytesseract_ocr import PytesseractOCR
from src.ocr_tools.paddleocr_tool import PaddleOCRTool
from src.ocr_tools.easyocr_tool import EasyOCRTool

class OCRBenchmark:
    """Benchmark multiple OCR tools on deed PDFs"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.ocr_tools = self._initialize_ocr_tools()
        self.results = []
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration for OCR tools"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            "tesseract": {
                "tesseract_config": "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()[]{}\"'-/ "
            },
            "paddleocr": {
                "use_angle_cls": True,
                "lang": "en",
                "show_log": False
            },
            "easyocr": {
                "languages": ["en"],
                "use_gpu": True
            }
        }
    
    def _initialize_ocr_tools(self) -> Dict[str, Any]:
        """Initialize all OCR tools"""
        tools = {}
        
        try:
            tools['tesseract'] = PytesseractOCR(self.config.get('tesseract', {}))
        except Exception as e:
            print(f"Failed to initialize Tesseract: {e}")
        
        try:
            tools['paddleocr'] = PaddleOCRTool(self.config.get('paddleocr', {}))
        except Exception as e:
            print(f"Failed to initialize PaddleOCR: {e}")
        
        try:
            tools['easyocr'] = EasyOCRTool(self.config.get('easyocr', {}))
        except Exception as e:
            print(f"Failed to initialize EasyOCR: {e}")
        
        return tools
    
    def process_single_pdf(self, pdf_path: str, tool_name: str) -> Dict[str, Any]:
        """Process a single PDF with a specific OCR tool"""
        if tool_name not in self.ocr_tools:
            return {
                'pdf_path': pdf_path,
                'tool': tool_name,
                'error': f'Tool {tool_name} not available'
            }
        
        try:
            result = self.ocr_tools[tool_name].extract_text(pdf_path)
            result['pdf_path'] = pdf_path
            result['tool'] = tool_name
            result['pdf_name'] = Path(pdf_path).name
            return result
        except Exception as e:
            return {
                'pdf_path': pdf_path,
                'tool': tool_name,
                'error': str(e),
                'text': '',
                'confidence': 0.0,
                'processing_time': 0.0
            }
    
    def run_benchmark(self, pdf_directory: str, output_dir: str, max_workers: int = 3):
        """Run OCR benchmark on all PDFs in directory"""
        pdf_dir = Path(pdf_directory)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all PDF files
        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")
        
        # Create tasks for all combinations of PDFs and OCR tools
        tasks = []
        for pdf_file in pdf_files:
            for tool_name in self.ocr_tools.keys():
                tasks.append((str(pdf_file), tool_name))
        
        print(f"Running {len(tasks)} OCR tasks...")
        
        # Process with thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self.process_single_pdf, pdf_path, tool_name): (pdf_path, tool_name)
                for pdf_path, tool_name in tasks
            }
            
            for future in as_completed(future_to_task):
                pdf_path, tool_name = future_to_task[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    
                    # Save individual result
                    pdf_name = Path(pdf_path).stem
                    result_file = output_path / f"{pdf_name}_{tool_name}.json"
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    
                    print(f"Completed: {pdf_name} with {tool_name}")
                    
                except Exception as e:
                    print(f"Error processing {pdf_path} with {tool_name}: {e}")
        
        # Save consolidated results
        self._save_benchmark_results(output_path)
        self._generate_comparison_report(output_path)
    
    def _save_benchmark_results(self, output_path: Path):
        """Save all results to JSON and CSV"""
        # Save raw results
        with open(output_path / "all_results.json", 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Create summary DataFrame
        summary_data = []
        for result in self.results:
            summary_data.append({
                'pdf_name': result.get('pdf_name', ''),
                'tool': result.get('tool', ''),
                'confidence': result.get('confidence', 0),
                'processing_time': result.get('processing_time', 0),
                'text_length': len(result.get('text', '')),
                'has_error': 'error' in result
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path / "benchmark_summary.csv", index=False)
    
    def _generate_comparison_report(self, output_path: Path):
        """Generate a comparison report"""
        df = pd.read_csv(output_path / "benchmark_summary.csv")
        
        report = {
            'summary_by_tool': {},
            'summary_by_pdf': {},
            'overall_stats': {}
        }
        
        # Stats by tool
        for tool in df['tool'].unique():
            tool_data = df[df['tool'] == tool]
            report['summary_by_tool'][tool] = {
                'avg_confidence': tool_data['confidence'].mean(),
                'avg_processing_time': tool_data['processing_time'].mean(),
                'avg_text_length': tool_data['text_length'].mean(),
                'error_rate': tool_data['has_error'].mean(),
                'total_processed': len(tool_data)
            }
        
        # Stats by PDF
        for pdf in df['pdf_name'].unique():
            pdf_data = df[df['pdf_name'] == pdf]
            report['summary_by_pdf'][pdf] = {
                'best_confidence_tool': pdf_data.loc[pdf_data['confidence'].idxmax(), 'tool'],
                'fastest_tool': pdf_data.loc[pdf_data['processing_time'].idxmin(), 'tool'],
                'longest_text_tool': pdf_data.loc[pdf_data['text_length'].idxmax(), 'tool']
            }
        
        # Overall stats
        report['overall_stats'] = {
            'total_pdfs': df['pdf_name'].nunique(),
            'total_tools': df['tool'].nunique(),
            'avg_confidence_all': df['confidence'].mean(),
            'avg_processing_time_all': df['processing_time'].mean()
        }
        
        with open(output_path / "comparison_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Benchmark complete! Results saved to {output_path}") 