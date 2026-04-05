#!/usr/bin/env python3
"""
Comprehensive Oil and Gas Classification Test Suite
=================================================

This script thoroughly tests the core classification logic on the entire dataset
with focus on:
1. High specificity for no-reservations (detecting true negatives)
2. Overall performance metrics
3. Error analysis and diagnostic reporting
4. Comparison with previous baselines

Usage:
    python test_core_classification.py
"""

import os
import sys
import json
import time
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import traceback

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the processor (same class used by the deployed app)
try:
    from src.mineral_rights.document_classifier import DocumentProcessor
    print("✅ Successfully imported DocumentProcessor")
except ImportError as e1:
    print(f"❌ Direct import failed: {e1}")
    try:
        sys.path.insert(0, str(project_root / "src"))
        from mineral_rights.document_classifier import DocumentProcessor
        print("✅ Successfully imported DocumentProcessor (method 2)")
    except ImportError as e2:
        print(f"❌ Module import failed: {e2}")
        try:
            import importlib.util
            classifier_path = project_root / "src" / "mineral_rights" / "document_classifier.py"
            spec = importlib.util.spec_from_file_location("document_classifier", classifier_path)
            document_classifier = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(document_classifier)
            DocumentProcessor = document_classifier.DocumentProcessor
            print("✅ Successfully imported DocumentProcessor (direct file import)")
        except Exception as e3:
            print(f"❌ All import methods failed:")
            print(f"   Method 1: {e1}")
            print(f"   Method 2: {e2}")
            print(f"   Method 3: {e3}")
            print(f"\nDebugging info:")
            print(f"   Project root: {project_root}")
            print(f"   Classifier file exists: {(project_root / 'src' / 'mineral_rights' / 'document_classifier.py').exists()}")
            print(f"   Current working directory: {os.getcwd()}")
            print(f"   Python path: {sys.path[:3]}...")
            sys.exit(1)


@dataclass
class TestResult:
    """Individual document test result"""
    filename: str
    true_label: int  # 0 = no-reservs, 1 = reservs
    predicted_label: int
    confidence: float
    processing_time: float
    samples_used: int
    early_stopped: bool
    stopped_at_page: Optional[int]
    pages_processed: int
    error: Optional[str] = None
    # Reasoning from first sample on the deciding page (for diagnosis)
    model_reasoning: Optional[str] = None
    ocr_text_preview: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for the classifier"""
    # Basic metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float  # True Negative Rate (critical for no-reservations)
    
    # Confusion matrix
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    # Processing stats
    avg_confidence: float
    avg_processing_time: float
    avg_samples_used: float
    total_processing_time: float
    
    # Error analysis
    false_positive_rate: float
    false_negative_rate: float
    misclassification_rate: float


class CoreClassificationTester:
    """Comprehensive testing framework for oil and gas classification"""

    def __init__(self, api_key: str):
        # Use DocumentProcessor — the same class the deployed app uses
        self.processor = DocumentProcessor(api_key=api_key)
        self.results: List[TestResult] = []
        self.start_time = datetime.now()
        
    def get_dataset_files(self) -> Tuple[List[Path], List[Path]]:
        """Get all PDF files from reservs and no-reservs directories"""
        # Look for data directory relative to project root
        data_dir = project_root / "data"
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found at {data_dir}")
        
        reservs_files = list((data_dir / "reservs").glob("*.pdf"))
        no_reservs_files = list((data_dir / "no-reservs").glob("*.pdf"))
        
        print(f"📁 Dataset Overview:")
        print(f"   • Reservations (positive): {len(reservs_files)} files")
        print(f"   • No-reservations (negative): {len(no_reservs_files)} files")
        print(f"   • Total: {len(reservs_files) + len(no_reservs_files)} files")
        
        return reservs_files, no_reservs_files
    
    def test_single_document(
        self,
        pdf_path: Path,
        true_label: int,
        max_samples: int = 6,
        confidence_threshold: float = 0.7
    ) -> TestResult:
        """Test classification on a single document using the same parameters as the deployed app."""
        start_time = time.time()

        try:
            # Exact same call as app.py /predict endpoint
            result = self.processor.process_document(
                str(pdf_path),
                max_samples=max_samples,
                confidence_threshold=confidence_threshold,
                page_strategy="first_few",
                high_recall_mode=True,
            )

            processing_time = time.time() - start_time

            # Pull reasoning + OCR from the deciding page's first sample
            reasoning = None
            ocr_preview = None
            chunk_analysis = result.get('chunk_analysis', [])
            detailed_samples = result.get('detailed_samples', [])
            if detailed_samples:
                s = detailed_samples[0]
                reasoning = s.get('raw_response') or s.get('reasoning')
            elif chunk_analysis:
                # Fallback: grab from chunk_analysis if detailed_samples not populated
                for chunk in chunk_analysis:
                    if chunk.get('samples'):
                        reasoning = chunk['samples'][0].get('reasoning')
                        break
            ocr_text = result.get('ocr_text', '')
            if ocr_text:
                ocr_preview = ocr_text[:800]

            return TestResult(
                filename=pdf_path.name,
                true_label=true_label,
                predicted_label=result['classification'],
                confidence=result['confidence'],
                processing_time=processing_time,
                samples_used=result.get('samples_used', 0),
                early_stopped=result.get('early_stopped', False),
                stopped_at_page=result.get('stopped_at_chunk'),
                pages_processed=result.get('pages_processed', 0),
                model_reasoning=reasoning,
                ocr_text_preview=ocr_preview,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            return TestResult(
                filename=pdf_path.name,
                true_label=true_label,
                predicted_label=-1,
                confidence=0.0,
                processing_time=processing_time,
                samples_used=0,
                early_stopped=False,
                stopped_at_page=None,
                pages_processed=0,
                error=error_msg
            )
    
    def run_full_evaluation(
        self, 
        max_samples: int = 8,
        confidence_threshold: float = 0.80,
        verbose: bool = True
    ) -> PerformanceMetrics:
        """Run evaluation on entire dataset"""
        
        print(f"🚀 Starting Comprehensive Classification Test")
        print(f"{'='*60}")
        print(f"Parameters:")
        print(f"   • Max samples per document: {max_samples}")
        print(f"   • Confidence threshold: {confidence_threshold}")
        print(f"   • Focus: HIGH SPECIFICITY (detecting no-reservations)")
        print()
        
        reservs_files, no_reservs_files = self.get_dataset_files()
        
        # Test no-reservations first (priority)
        print("🔍 Testing NO-RESERVATIONS documents (priority)...")
        for i, pdf_path in enumerate(no_reservs_files, 1):
            if verbose:
                print(f"   [{i:2d}/{len(no_reservs_files)}] {pdf_path.name}", end=" ... ")
            
            result = self.test_single_document(pdf_path, true_label=0, 
                                             max_samples=max_samples, 
                                             confidence_threshold=confidence_threshold)
            self.results.append(result)
            
            if verbose:
                if result.error:
                    print(f"❌ ERROR: {result.error}")
                else:
                    status = "✅ CORRECT" if result.predicted_label == 0 else "❌ FALSE POSITIVE"
                    print(f"{status} (conf: {result.confidence:.3f}, time: {result.processing_time:.1f}s)")
        
        print(f"\n🔥 Testing RESERVATIONS documents...")
        for i, pdf_path in enumerate(reservs_files, 1):
            if verbose:
                print(f"   [{i:2d}/{len(reservs_files)}] {pdf_path.name}", end=" ... ")
            
            result = self.test_single_document(pdf_path, true_label=1, 
                                             max_samples=max_samples, 
                                             confidence_threshold=confidence_threshold)
            self.results.append(result)
            
            if verbose:
                if result.error:
                    print(f"❌ ERROR: {result.error}")
                else:
                    status = "✅ CORRECT" if result.predicted_label == 1 else "❌ FALSE NEGATIVE"
                    print(f"{status} (conf: {result.confidence:.3f}, time: {result.processing_time:.1f}s)")
        
        # Calculate performance metrics
        metrics = self.calculate_metrics()
        
        print(f"\n{'='*60}")
        print(f"🎯 EVALUATION COMPLETE")
        print(f"   Total processing time: {metrics.total_processing_time:.1f} seconds")
        print(f"   Average time per document: {metrics.avg_processing_time:.1f} seconds")
        
        return metrics
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        # Filter out error cases
        valid_results = [r for r in self.results if r.error is None]
        
        if not valid_results:
            raise ValueError("No valid results to calculate metrics")
        
        # Confusion matrix
        tp = sum(1 for r in valid_results if r.true_label == 1 and r.predicted_label == 1)
        fp = sum(1 for r in valid_results if r.true_label == 0 and r.predicted_label == 1)
        tn = sum(1 for r in valid_results if r.true_label == 0 and r.predicted_label == 0)
        fn = sum(1 for r in valid_results if r.true_label == 1 and r.predicted_label == 0)
        
        # Basic metrics
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Critical metric
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Error rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        # Processing stats
        confidences = [r.confidence for r in valid_results]
        times = [r.processing_time for r in valid_results]
        samples = [r.samples_used for r in valid_results]
        
        return PerformanceMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            specificity=specificity,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            avg_confidence=statistics.mean(confidences) if confidences else 0,
            avg_processing_time=statistics.mean(times) if times else 0,
            avg_samples_used=statistics.mean(samples) if samples else 0,
            total_processing_time=sum(times),
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            misclassification_rate=1 - accuracy
        )
    
    def generate_detailed_report(self, metrics: PerformanceMetrics) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("COMPREHENSIVE OIL & GAS CLASSIFICATION TEST REPORT")
        report.append("=" * 60)
        report.append(f"Test Date: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Documents Tested: {len(self.results)}")
        report.append("")
        
        # Performance Summary
        report.append("🎯 PERFORMANCE SUMMARY")
        report.append("-" * 30)
        report.append(f"Overall Accuracy:     {metrics.accuracy:.1%}")
        report.append(f"Specificity (TNR):    {metrics.specificity:.1%} ⭐ KEY METRIC")
        report.append(f"Sensitivity (Recall): {metrics.recall:.1%}")
        report.append(f"Precision:            {metrics.precision:.1%}")
        report.append(f"F1-Score:            {metrics.f1_score:.1%}")
        report.append("")
        
        # Error Analysis (Critical for your use case)
        report.append("🚨 ERROR ANALYSIS (Focus: No-Reservations)")
        report.append("-" * 40)
        report.append(f"False Positive Rate:  {metrics.false_positive_rate:.1%} (wrongly flagged as reservations)")
        report.append(f"False Negative Rate:  {metrics.false_negative_rate:.1%} (missed actual reservations)")
        report.append(f"Misclassification:    {metrics.misclassification_rate:.1%}")
        report.append("")
        
        # Confusion Matrix
        report.append("📊 CONFUSION MATRIX")
        report.append("-" * 20)
        report.append("                    Predicted")
        report.append("                No-Res  Reserv")
        report.append(f"Actual  No-Res    {metrics.true_negatives:3d}     {metrics.false_positives:3d}")
        report.append(f"        Reserv    {metrics.false_negatives:3d}     {metrics.true_positives:3d}")
        report.append("")
        
        # Processing Stats
        report.append("⚡ PROCESSING STATISTICS")
        report.append("-" * 25)
        report.append(f"Total Processing Time:    {metrics.total_processing_time:.1f} seconds")
        report.append(f"Average Time per Doc:     {metrics.avg_processing_time:.1f} seconds")
        report.append(f"Average Confidence:       {metrics.avg_confidence:.3f}")
        report.append(f"Average Samples Used:     {metrics.avg_samples_used:.1f}")
        report.append("")
        
        # Error Details
        error_results = [r for r in self.results if r.error is not None]
        if error_results:
            report.append("❌ PROCESSING ERRORS")
            report.append("-" * 18)
            for err_result in error_results:
                report.append(f"• {err_result.filename}: {err_result.error}")
            report.append("")
        
        # Misclassification Details
        fp_results = [r for r in self.results if r.error is None and r.true_label == 0 and r.predicted_label == 1]
        fn_results = [r for r in self.results if r.error is None and r.true_label == 1 and r.predicted_label == 0]
        
        if fp_results:
            report.append("🔍 FALSE POSITIVES (Critical - No-reservs wrongly flagged)")
            report.append("-" * 55)
            for fp in fp_results:
                report.append(f"• {fp.filename} (confidence: {fp.confidence:.3f})")
            report.append("")
        
        if fn_results:
            report.append("🔍 FALSE NEGATIVES (Missed actual reservations)")
            report.append("-" * 45)
            for fn in fn_results:
                report.append(f"• {fn.filename} (confidence: {fn.confidence:.3f})")
            report.append("")
        
        # Benchmark Comparison
        report.append("📈 BENCHMARK COMPARISON")
        report.append("-" * 23)
        report.append("Target Metrics for Production:")
        report.append(f"  • Specificity (TNR):  ≥95% [Current: {metrics.specificity:.1%}]")
        report.append(f"  • Overall Accuracy:   ≥90% [Current: {metrics.accuracy:.1%}]")
        report.append(f"  • False Positive Rate: ≤5% [Current: {metrics.false_positive_rate:.1%}]")
        
        # Recommendations
        report.append("")
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 17)
        if metrics.specificity < 0.95:
            report.append("⚠️  Specificity below target - tune confidence threshold higher")
        if metrics.false_positive_rate > 0.05:
            report.append("⚠️  Too many false positives - review conservative prompt")
        if metrics.accuracy < 0.90:
            report.append("⚠️  Overall accuracy below target - investigate feature engineering")
        if metrics.specificity >= 0.95 and metrics.accuracy >= 0.90:
            report.append("✅ Performance meets production targets!")
        
        return "\n".join(report)
    
    def save_results(self, metrics: PerformanceMetrics, output_dir: str = "test_results"):
        """Save detailed results and report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        error_count = sum(1 for r in self.results if r.error is not None)

        results_data = {
            "timestamp": self.start_time.isoformat(),
            "metrics": asdict(metrics),
            "detailed_results": [asdict(r) for r in self.results],
            "parameters": {
                "focus": "Matches deployed app exactly",
                "max_samples": 6,
                "confidence_threshold": 0.7,
                "page_strategy": "first_few",
                "high_recall_mode": True,
                "total_documents": len(self.results),
                "errored_documents": error_count,
                "complete": error_count == 0,
            }
        }

        json_path = output_path / f"core_classification_test_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        report = self.generate_detailed_report(metrics)
        report_path = output_path / f"core_classification_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"\n💾 Results saved:")
        print(f"   • Detailed data: {json_path}")
        print(f"   • Human report:  {report_path}")
        if error_count > 0:
            print(f"\n⚠️  {error_count} document(s) errored — metrics are incomplete.")
            print(f"   Top up API credits, then run:")
            print(f"   ANTHROPIC_API_KEY=... python scripts/test_core_classification.py --resume {json_path}")

        return json_path


def resume_from(json_path: str, api_key: str):
    """Re-run only errored documents from a previous run and merge results."""
    with open(json_path) as f:
        previous = json.load(f)

    prior_results = previous["detailed_results"]
    errored = [r for r in prior_results if r.get("error") is not None]
    good    = [r for r in prior_results if r.get("error") is None]

    if not errored:
        print("✅ No errored documents in that run — nothing to resume.")
        return

    print(f"🔁 RESUME MODE — re-running {len(errored)} errored document(s)")
    print(f"   Keeping {len(good)} already-good result(s) from previous run.")
    print()

    tester = CoreClassificationTester(api_key)

    for entry in errored:
        filename  = entry["filename"]
        true_label = entry["true_label"]
        # Locate the file in the dataset
        candidates = list(project_root.rglob(filename))
        if not candidates:
            print(f"   ⚠️  File not found on disk: {filename} — skipping")
            # Keep the error record so the count stays accurate
            from dataclasses import fields as dc_fields
            tester.results.append(TestResult(**{k: entry[k] for k in entry}))
            continue

        pdf_path = candidates[0]
        print(f"   Re-running: {filename}", end=" ... ")
        result = tester.test_single_document(pdf_path, true_label=true_label)
        tester.results.append(result)
        if result.error:
            print(f"❌ STILL ERRORING: {result.error[:80]}")
        else:
            status = "✅ CORRECT" if result.predicted_label == result.true_label else "❌ WRONG"
            print(f"{status} (conf: {result.confidence:.3f})")

    # Merge with the good results from the previous run
    for entry in good:
        tester.results.append(TestResult(**{k: entry[k] for k in entry}))

    # Recompute metrics over all 90 documents
    remaining_errors = sum(1 for r in tester.results if r.error is not None)
    if remaining_errors > 0:
        print(f"\n⚠️  {remaining_errors} document(s) still erroring — metrics still incomplete.")
    else:
        print(f"\n✅ All documents resolved — computing final metrics on full dataset.")

    metrics = tester.calculate_metrics()
    print(f"\n{tester.generate_detailed_report(metrics)}")
    tester.save_results(metrics)

    print(f"\n🏆 FINAL SUMMARY ({len(tester.results)} documents):")
    print(f"   Accuracy:          {metrics.accuracy:.1%}")
    print(f"   Specificity (TNR): {metrics.specificity:.1%}")
    print(f"   False Positives:   {metrics.false_positives}")
    print(f"   False Negatives:   {metrics.false_negatives}")


def diagnose_failures(json_path: str, api_key: str):
    """Re-run only misclassified docs and print full model reasoning for diagnosis."""
    with open(json_path) as f:
        previous = json.load(f)

    prior_results = previous["detailed_results"]
    misclassified = [
        r for r in prior_results
        if r.get('error') is None
        and r['true_label'] != r['predicted_label']
    ]

    if not misclassified:
        print("✅ No misclassified documents in that file.")
        return

    fps = [r for r in misclassified if r['true_label'] == 0]
    fns = [r for r in misclassified if r['true_label'] == 1]
    print(f"🔬 DIAGNOSE MODE — re-running {len(misclassified)} misclassified docs")
    print(f"   {len(fps)} false positives + {len(fns)} false negatives")
    print()

    tester = CoreClassificationTester(api_key)

    for entry in misclassified:
        filename = entry['filename']
        true_label = entry['true_label']
        label_str = "NO-RESERV (should be 0)" if true_label == 0 else "HAS-RESERV (should be 1)"
        error_type = "FALSE POSITIVE" if true_label == 0 else "FALSE NEGATIVE"

        candidates = list(project_root.rglob(filename))
        if not candidates:
            print(f"⚠️  {filename}: not found on disk — skipping")
            continue

        pdf_path = candidates[0]
        print(f"\n{'='*70}")
        print(f"🔬 [{error_type}] {filename}")
        print(f"   True label: {label_str}")
        print(f"   Running...", flush=True)

        result = tester.test_single_document(pdf_path, true_label=true_label)

        outcome = "✅ NOW CORRECT" if result.predicted_label == true_label else "❌ STILL WRONG"
        print(f"   {outcome} — predicted={result.predicted_label}, conf={result.confidence:.3f}")
        print()

        if result.ocr_text_preview:
            print(f"📄 OCR TEXT (first 800 chars):")
            print(result.ocr_text_preview)
            print()

        if result.model_reasoning:
            print(f"🤖 MODEL REASONING:")
            print(result.model_reasoning[:1500])
        else:
            print("⚠️  No reasoning captured (check detailed_samples in result)")

    print(f"\n{'='*70}")
    print("🔬 Diagnose run complete. Use findings to refine the prompt.")


def main():
    """Main test execution"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", metavar="JSON_FILE",
                        help="Path to a previous results JSON — re-run only errored docs and merge")
    parser.add_argument("--diagnose", metavar="JSON_FILE",
                        help="Re-run only misclassified docs from a results JSON and print full reasoning")
    args = parser.parse_args()

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        return

    try:
        if args.resume:
            resume_from(args.resume, api_key)
            return

        if args.diagnose:
            diagnose_failures(args.diagnose, api_key)
            return

        print("🧪 CORE CLASSIFICATION TESTING SUITE")
        print("=" * 50)
        print("🎯 PRIORITY: High specificity for no-reservations")
        print("📊 Testing entire dataset for comprehensive evaluation")
        print(f"📁 Project root: {project_root}")
        print()

        tester = CoreClassificationTester(api_key)

        metrics = tester.run_full_evaluation(
            max_samples=6,
            confidence_threshold=0.7,
            verbose=True
        )

        print(f"\n{tester.generate_detailed_report(metrics)}")
        json_path = tester.save_results(metrics)

        print(f"\n🏆 QUICK SUMMARY:")
        print(f"   Accuracy:          {metrics.accuracy:.1%}")
        print(f"   Specificity (TNR): {metrics.specificity:.1%}")
        print(f"   False Positives:   {metrics.false_positives} documents")
        print(f"   Processing Time:   {metrics.total_processing_time:.1f}s")

        if metrics.specificity >= 0.95 and metrics.accuracy >= 0.90:
            print(f"\n✅ CORE CLASSIFICATION ABILITY: EXCELLENT")
        elif metrics.specificity >= 0.90:
            print(f"\n⚠️  CORE CLASSIFICATION ABILITY: GOOD (room for improvement)")
        else:
            print(f"\n❌ CORE CLASSIFICATION ABILITY: NEEDS ATTENTION")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
