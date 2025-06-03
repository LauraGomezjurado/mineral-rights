#!/usr/bin/env python3
"""
Test Accuracy Script
===================

Test the classification accuracy on a small subset of documents.
Now with chunk-by-chunk early stopping as the default method.
"""

import json
from pathlib import Path
from document_classifier import DocumentProcessor

def test_accuracy_sample():
    """Test accuracy on a small sample of documents with early stopping"""
    
    processor = DocumentProcessor()
    
    # Test with a few documents from each category
    test_files = [
        # From reservs folder (should be classified as 1)
        ("data/reservs/Washington DB 475_646 - 4.23.2025.pdf", 1),
        ("data/reservs/Allegheny DB 2470_179 - 4.23.2025.pdf", 1),
        
        # From no-reservs folder (should be classified as 0)  
        ("data/no-reservs/Allegheny 1910-768.pdf", 0),
        ("data/no-reservs/Allegheny 8277_16 - 4.23.2025.pdf", 0),
    ]
    
    # Test different processing strategies
    strategies = [
        {
            "name": "Early Stopping (Default)",
            "page_strategy": "sequential_early_stop",  # New default
            "max_tokens_per_page": 8000,
            "combine_method": "early_stop"
        },
        # Uncomment to test other strategies:
        # {
        #     "name": "Enhanced Multi-Page",
        #     "page_strategy": "first_few",  # First 3 pages
        #     "max_tokens_per_page": 8000,
        #     "combine_method": "concatenate"
        # },
        # {
        #     "name": "Legacy Single Page",
        #     "page_strategy": "first_only",
        #     "max_tokens_per_page": 8000,
        #     "combine_method": "concatenate"
        # },
        # {
        #     "name": "All Pages",
        #     "page_strategy": "all",
        #     "max_tokens_per_page": 6000,  # Lower per page for many pages
        #     "combine_method": "summarize"
        # }
    ]
    
    all_results = {}
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"TESTING WITH STRATEGY: {strategy['name']}")
        print(f"{'='*60}")
        
        results = []
        correct_count = 0
        
        for i, (file_path, expected_label) in enumerate(test_files, 1):
            file_path = Path(file_path)
            
            if not file_path.exists():
                print(f"[{i}] SKIP: {file_path.name} (file not found)")
                continue
                
            print(f"\n[{i}] Processing: {file_path.name}")
            print(f"Expected: {expected_label} ({'Has Reservations' if expected_label == 1 else 'No Reservations'})")
            
            try:
                # Process with the specified strategy
                result = processor.process_document(
                    str(file_path), 
                    max_samples=5, 
                    confidence_threshold=0.6,
                    page_strategy=strategy["page_strategy"],
                    max_tokens_per_page=strategy["max_tokens_per_page"],
                    combine_method=strategy["combine_method"]
                )
                
                predicted_label = result['classification']
                confidence = result['confidence']
                is_correct = (predicted_label == expected_label)
                
                if is_correct:
                    correct_count += 1
                
                print(f"Pages processed: {result['pages_processed']}")
                if 'total_pages_in_document' in result:
                    print(f"Total pages in document: {result['total_pages_in_document']}")
                    if result.get('stopped_at_chunk'):
                        print(f"🎯 Stopped early at page: {result['stopped_at_chunk']}")
                        saved_pages = result['total_pages_in_document'] - result['pages_processed']
                        print(f"⚡ Efficiency: Saved {saved_pages} pages")
                
                print(f"OCR text length: {result['ocr_text_length']} characters")
                print(f"Predicted: {predicted_label} ({'Has Reservations' if predicted_label == 1 else 'No Reservations'})")
                print(f"Final Confidence: {confidence:.3f}")
                print(f"Votes: {result['votes']}")
                
                # Show chunk analysis if available (for early stopping strategy)
                if 'chunk_analysis' in result and result['chunk_analysis']:
                    print("Chunk-by-chunk analysis:")
                    for chunk in result['chunk_analysis']:
                        page_num = chunk['page_number']
                        chunk_class = chunk['classification']
                        chunk_conf = chunk['confidence']
                        status = "🎯 FOUND!" if chunk_class == 1 else "✓ None"
                        print(f"  Page {page_num}: {status} (conf: {chunk_conf:.3f})")
                        if chunk_class == 1:
                            break
                
                # Show individual sample confidence scores for detailed samples
                if 'detailed_samples' in result and result['detailed_samples']:
                    print("Individual Sample Details:")
                    for j, sample in enumerate(result['detailed_samples'], 1):
                        print(f"  Sample {j}: Class={sample['predicted_class']}, Individual Confidence={sample['confidence_score']:.3f}")
                
                print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
                
                # Save comprehensive result
                result_data = {
                    'file': file_path.name,
                    'document_path': result['document_path'],
                    'expected': expected_label,
                    'predicted': predicted_label,
                    'confidence': confidence,
                    'correct': is_correct,
                    'samples_used': result['samples_used'],
                    'early_stopped': result['early_stopped'],
                    'votes': result['votes'],
                    'pages_processed': result['pages_processed'],
                    'page_strategy': result['page_strategy'],
                    'max_tokens_per_page': result['max_tokens_per_page'],
                    'ocr_text_length': result['ocr_text_length'],
                    'ocr_text': result['ocr_text'],
                    'detailed_samples': result.get('detailed_samples', [])
                }
                
                # Add early stopping specific fields if available
                if 'total_pages_in_document' in result:
                    result_data.update({
                        'total_pages_in_document': result['total_pages_in_document'],
                        'stopped_at_chunk': result.get('stopped_at_chunk'),
                        'chunk_analysis': result.get('chunk_analysis', []),
                        'efficiency_ratio': result['pages_processed'] / result['total_pages_in_document']
                    })
                
                results.append(result_data)
                
            except Exception as e:
                print(f"ERROR: {e}")
                continue
        
        # Summary for this strategy
        total_tested = len(results)
        accuracy = correct_count / total_tested if total_tested > 0 else 0
        
        print(f"\n" + "=" * 50)
        print(f"SUMMARY FOR {strategy['name']}")
        print(f"=" * 50)
        print(f"Total tested: {total_tested}")
        print(f"Correct: {correct_count}")
        print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        if results:
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            avg_text_length = sum(r['ocr_text_length'] for r in results) / len(results)
            avg_pages = sum(r['pages_processed'] for r in results) / len(results)
            
            print(f"Average final confidence: {avg_confidence:.3f}")
            print(f"Average OCR text length: {avg_text_length:.0f} characters")
            print(f"Average pages processed: {avg_pages:.1f}")
            
            # Early stopping specific metrics
            early_stopping_results = [r for r in results if 'efficiency_ratio' in r]
            if early_stopping_results:
                avg_efficiency = sum(r['efficiency_ratio'] for r in early_stopping_results) / len(early_stopping_results)
                early_stops = sum(1 for r in early_stopping_results if r.get('stopped_at_chunk'))
                print(f"Average efficiency: {avg_efficiency:.3f} (pages processed / total pages)")
                print(f"Early stops: {early_stops}/{len(early_stopping_results)}")
            
            # Show detailed results
            print(f"\nDetailed Results:")
            for r in results:
                status = "✅" if r['correct'] else "❌"
                if 'efficiency_ratio' in r:
                    early_indicator = "🎯" if r.get('stopped_at_chunk') else "📄"
                    efficiency_info = f" ({r['pages_processed']}/{r.get('total_pages_in_document', '?')} pages)"
                else:
                    early_indicator = ""
                    efficiency_info = f" ({r['pages_processed']} pages)"
                
                print(f"  {status} {early_indicator} {r['file']}: {r['expected']} → {r['predicted']} (conf: {r['confidence']:.3f}{efficiency_info})")
        
        all_results[strategy['name']] = {
            'strategy_config': strategy,
            'summary': {
                'total_tested': total_tested,
                'correct': correct_count,
                'accuracy': accuracy,
                'average_confidence': avg_confidence if results else 0,
                'average_text_length': avg_text_length if results else 0,
                'average_pages_processed': avg_pages if results else 0
            },
            'detailed_results': results
        }
        
        # Add early stopping metrics if available
        if early_stopping_results:
            all_results[strategy['name']]['summary'].update({
                'average_efficiency': avg_efficiency,
                'early_stops': early_stops,
                'total_with_efficiency_data': len(early_stopping_results)
            })
    
    # Save comprehensive results
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save all strategies comparison
    with open(output_dir / "test_accuracy_with_early_stopping.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Save detailed results for the main strategy
    main_strategy = list(all_results.keys())[0]
    main_results = all_results[main_strategy]['detailed_results']
    
    with open(output_dir / "test_accuracy_results_detailed.json", "w") as f:
        json.dump({
            'strategy_used': main_strategy,
            'summary': all_results[main_strategy]['summary'],
            'detailed_results': main_results
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    
    for strategy_name, strategy_results in all_results.items():
        summary = strategy_results['summary']
        print(f"{strategy_name}:")
        print(f"  Accuracy: {summary['accuracy']:.3f} ({summary['accuracy']*100:.1f}%)")
        print(f"  Avg Text Length: {summary['average_text_length']:.0f} chars")
        print(f"  Avg Pages: {summary['average_pages_processed']:.1f}")
        if 'average_efficiency' in summary:
            print(f"  Avg Efficiency: {summary['average_efficiency']:.3f}")
            print(f"  Early Stops: {summary['early_stops']}/{summary['total_with_efficiency_data']}")
    
    print(f"\nResults saved to:")
    print(f"  - test_results/test_accuracy_with_early_stopping.json (all strategies)")
    print(f"  - test_results/test_accuracy_results_detailed.json (main strategy detailed)")
    
    return all_results

if __name__ == "__main__":
    test_accuracy_sample() 