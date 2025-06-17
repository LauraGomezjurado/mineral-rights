#!/usr/bin/env python3
"""
False Positive Analysis for Mineral Rights Classification
========================================================

Analyze documents that were wrongly classified as having reservations
to identify patterns and improve specificity.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Set
from collections import Counter
from document_classifier import DocumentProcessor

def analyze_false_positives():
    """Analyze false positive documents to identify common patterns"""
    
    print("🔍 FALSE POSITIVE ANALYSIS")
    print("=" * 60)
    
    # List of false positive documents from the evaluation
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
    
    print(f"Analyzing {len(false_positive_docs)} false positive documents...")
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Extract text and analyze each document
    all_texts = []
    all_keywords = []
    all_phrases = []
    
    for i, doc_path in enumerate(false_positive_docs, 1):
        print(f"\n[{i}/{len(false_positive_docs)}] Analyzing: {Path(doc_path).name}")
        
        try:
            # Process document to get OCR text (just first page for speed)
            result = processor.process_document(
                doc_path,
                max_samples=1,  # Just need the OCR text
                max_pages=1,    # First page only for analysis
                page_strategy="first_only"
            )
            
            ocr_text = result['ocr_text']
            print(f"  Extracted {len(ocr_text)} characters")
            
            all_texts.append({
                'document': Path(doc_path).name,
                'text': ocr_text,
                'classification': result['classification'],
                'confidence': result['confidence']
            })
            
            # Extract potential problematic keywords
            keywords = extract_reservation_keywords(ocr_text)
            all_keywords.extend(keywords)
            
            # Extract phrases around reservation keywords
            phrases = extract_context_phrases(ocr_text)
            all_phrases.extend(phrases)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Analyze patterns
    print(f"\n📊 PATTERN ANALYSIS")
    print("=" * 40)
    
    # Most common keywords
    keyword_counts = Counter(all_keywords)
    print(f"\n🔤 Most Common Keywords in False Positives:")
    for keyword, count in keyword_counts.most_common(10):
        print(f"  '{keyword}': {count} documents")
    
    # Most common phrases
    phrase_counts = Counter(all_phrases)
    print(f"\n📝 Most Common Phrases in False Positives:")
    for phrase, count in phrase_counts.most_common(15):
        print(f"  '{phrase}': {count} documents")
    
    # Save detailed analysis
    save_detailed_analysis(all_texts, keyword_counts, phrase_counts)
    
    # Generate improvement recommendations
    generate_recommendations(keyword_counts, phrase_counts)
    
    return all_texts

def extract_reservation_keywords(text: str) -> List[str]:
    """Extract reservation-related keywords from text"""
    
    keywords = []
    text_lower = text.lower()
    
    # Primary reservation keywords
    primary_keywords = [
        'reserves', 'reserved', 'reserving', 'reservation',
        'excepts', 'excepted', 'excepting', 'exception',
        'retains', 'retained', 'retaining',
        'coal', 'oil', 'gas', 'mineral', 'minerals', 'mining',
        'subject to', 'unto grantor', 'grantor reserves'
    ]
    
    for keyword in primary_keywords:
        if keyword in text_lower:
            keywords.append(keyword)
    
    return keywords

def extract_context_phrases(text: str, window_size: int = 50) -> List[str]:
    """Extract phrases around reservation keywords for context analysis"""
    
    phrases = []
    text_lower = text.lower()
    
    # Keywords that might indicate reservations
    trigger_words = ['reserves', 'excepts', 'retains', 'subject to', 'coal', 'oil', 'gas', 'mineral']
    
    for trigger in trigger_words:
        # Find all occurrences of the trigger word
        start = 0
        while True:
            pos = text_lower.find(trigger, start)
            if pos == -1:
                break
            
            # Extract context around the trigger word
            context_start = max(0, pos - window_size)
            context_end = min(len(text), pos + len(trigger) + window_size)
            context = text[context_start:context_end].strip()
            
            # Clean up the context
            context = re.sub(r'\s+', ' ', context)  # Normalize whitespace
            context = context.replace('\n', ' ')
            
            if len(context) > 20:  # Only keep substantial phrases
                phrases.append(context)
            
            start = pos + 1
    
    return phrases

def save_detailed_analysis(all_texts: List[Dict], keyword_counts: Counter, phrase_counts: Counter):
    """Save detailed analysis to files"""
    
    output_dir = Path("false_positive_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Save full text analysis
    with open(output_dir / "false_positive_texts.json", "w") as f:
        json.dump(all_texts, f, indent=2)
    
    # Save keyword analysis
    with open(output_dir / "keyword_analysis.json", "w") as f:
        json.dump(dict(keyword_counts), f, indent=2)
    
    # Save phrase analysis
    with open(output_dir / "phrase_analysis.json", "w") as f:
        json.dump(dict(phrase_counts), f, indent=2)
    
    print(f"\n💾 Detailed analysis saved to: {output_dir}")

def generate_recommendations(keyword_counts: Counter, phrase_counts: Counter) -> None:
    """Generate specific recommendations based on the analysis"""
    
    print(f"\n🎯 IMPROVEMENT RECOMMENDATIONS")
    print("=" * 50)
    
    # Analyze the most common problematic patterns
    top_keywords = [k for k, v in keyword_counts.most_common(5)]
    top_phrases = [p for p, v in phrase_counts.most_common(5)]
    
    print(f"\n1. 🔧 PROMPT ENHANCEMENT:")
    print(f"   Add specific examples of BOILERPLATE language to ignore:")
    
    for phrase in top_phrases[:3]:
        if any(word in phrase.lower() for word in ['subject to', 'matters of', 'record', 'otherwise']):
            print(f"   - \"{phrase[:80]}...\"")
    
    print(f"\n2. 📝 CLASSIFICATION LOGIC UPDATES:")
    print(f"   Enhance boilerplate detection for these patterns:")
    
    boilerplate_indicators = []
    for phrase in top_phrases:
        phrase_lower = phrase.lower()
        if any(indicator in phrase_lower for indicator in [
            'subject to', 'matters of record', 'restrictions of record',
            'does not enlarge', 'otherwise reserved', 'general warranty'
        ]):
            boilerplate_indicators.append(phrase[:60] + "...")
    
    for indicator in boilerplate_indicators[:5]:
        print(f"   - {indicator}")
    
    print(f"\n3. 🎚️  THRESHOLD ADJUSTMENTS:")
    print(f"   Consider raising confidence threshold from 0.5 to 0.7")
    print(f"   This would make the system more conservative about positive classifications")
    
    print(f"\n4. 🔍 FEATURE ENGINEERING:")
    print(f"   Enhance boilerplate_indicators feature to better detect:")
    for keyword in top_keywords[:5]:
        if keyword in ['subject to', 'matters of', 'record', 'otherwise']:
            print(f"   - Phrases containing '{keyword}'")

def main():
    """Main analysis function"""
    
    print("🚀 Starting False Positive Analysis...")
    
    try:
        results = analyze_false_positives()
        print(f"\n✅ Analysis complete! Found patterns in {len(results)} documents.")
        print("Check 'false_positive_analysis/' directory for detailed results.")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main() 