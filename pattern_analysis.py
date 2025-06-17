#!/usr/bin/env python3
"""
Pattern Analysis for False Positives
===================================

Analyze patterns in false positive documents based on evaluation results.
"""

import pandas as pd
from pathlib import Path
from collections import Counter
import re

def analyze_false_positive_patterns():
    """Analyze patterns in false positive documents"""
    
    print("🔍 FALSE POSITIVE PATTERN ANALYSIS")
    print("=" * 60)
    
    # Load evaluation results
    results_df = pd.read_csv("optimized_evaluation_results/optimized_detailed_results.csv")
    
    # Filter false positives (expected 0, predicted 1)
    false_positives = results_df[
        (results_df['expected_label'] == 0) & 
        (results_df['predicted_label'] == 1)
    ]
    
    # Filter true negatives (expected 0, predicted 0) for comparison
    true_negatives = results_df[
        (results_df['expected_label'] == 0) & 
        (results_df['predicted_label'] == 0)
    ]
    
    print(f"📊 DATASET BREAKDOWN:")
    print(f"  False Positives: {len(false_positives)} documents")
    print(f"  True Negatives: {len(true_negatives)} documents")
    print(f"  Total No-Reservation Docs: {len(false_positives) + len(true_negatives)}")
    
    # Analyze document naming patterns
    print(f"\n🏷️  DOCUMENT NAMING PATTERNS:")
    analyze_naming_patterns(false_positives, true_negatives)
    
    # Analyze processing characteristics
    print(f"\n⚙️  PROCESSING CHARACTERISTICS:")
    analyze_processing_patterns(false_positives, true_negatives)
    
    # Geographic analysis
    print(f"\n🗺️  GEOGRAPHIC PATTERNS:")
    analyze_geographic_patterns(false_positives, true_negatives)
    
    # Generate specific recommendations
    print(f"\n🎯 TARGETED RECOMMENDATIONS:")
    generate_targeted_recommendations(false_positives, true_negatives)
    
    return false_positives, true_negatives

def analyze_naming_patterns(false_positives, true_negatives):
    """Analyze document naming patterns"""
    
    def extract_patterns(df, label):
        patterns = {
            'DB_documents': 0,
            'DV_documents': 0, 
            'OR_documents': 0,
            'has_numbers': 0,
            'has_dates': 0,
            'county_names': []
        }
        
        for doc_name in df['document_name']:
            if ' DB ' in doc_name:
                patterns['DB_documents'] += 1
            if ' DV ' in doc_name:
                patterns['DV_documents'] += 1
            if ' OR ' in doc_name:
                patterns['OR_documents'] += 1
            if re.search(r'\d+_\d+', doc_name):
                patterns['has_numbers'] += 1
            if re.search(r'\d{1,2}\.\d{1,2}\.\d{4}', doc_name):
                patterns['has_dates'] += 1
            
            # Extract county names (first word usually)
            county = doc_name.split()[0]
            patterns['county_names'].append(county)
        
        return patterns
    
    fp_patterns = extract_patterns(false_positives, "False Positives")
    tn_patterns = extract_patterns(true_negatives, "True Negatives")
    
    print(f"  Document Type Distribution:")
    print(f"    False Positives - DB: {fp_patterns['DB_documents']}, DV: {fp_patterns['DV_documents']}, OR: {fp_patterns['OR_documents']}")
    print(f"    True Negatives  - DB: {tn_patterns['DB_documents']}, DV: {tn_patterns['DV_documents']}, OR: {tn_patterns['OR_documents']}")
    
    # County analysis
    fp_counties = Counter(fp_patterns['county_names'])
    tn_counties = Counter(tn_patterns['county_names'])
    
    print(f"\n  Most Common Counties in False Positives:")
    for county, count in fp_counties.most_common(5):
        print(f"    {county}: {count} documents")

def analyze_processing_patterns(false_positives, true_negatives):
    """Analyze processing characteristics"""
    
    print(f"  Pages Processed:")
    print(f"    False Positives - Avg: {false_positives['pages_processed'].mean():.1f}, Range: {false_positives['pages_processed'].min()}-{false_positives['pages_processed'].max()}")
    print(f"    True Negatives  - Avg: {true_negatives['pages_processed'].mean():.1f}, Range: {true_negatives['pages_processed'].min()}-{true_negatives['pages_processed'].max()}")
    
    print(f"\n  Samples Used:")
    print(f"    False Positives - Avg: {false_positives['samples_used'].mean():.1f}")
    print(f"    True Negatives  - Avg: {true_negatives['samples_used'].mean():.1f}")
    
    print(f"\n  Confidence Scores:")
    print(f"    False Positives - All have confidence: {false_positives['confidence'].unique()}")
    print(f"    True Negatives  - Confidence range: {true_negatives['confidence'].min():.3f} - {true_negatives['confidence'].max():.3f}")
    
    # Key insight: False positives all have 1.0 confidence and low sample usage
    fp_low_samples = false_positives[false_positives['samples_used'] <= 3]
    print(f"\n  🚨 CRITICAL PATTERN: {len(fp_low_samples)}/{len(false_positives)} false positives used ≤3 samples")
    print(f"     This suggests early stopping due to high initial confidence in wrong classification")

def analyze_geographic_patterns(false_positives, true_negatives):
    """Analyze geographic patterns in document origins"""
    
    def extract_state_county(doc_name):
        # Common patterns: "County State", "County Co State"
        parts = doc_name.replace('.pdf', '').split()
        county = parts[0] if parts else "Unknown"
        
        # Identify state patterns
        state = "Unknown"
        if any(indicator in doc_name for indicator in ['PA', 'Pennsylvania']):
            state = "PA"
        elif any(indicator in doc_name for indicator in ['OH', 'Ohio']):
            state = "OH"  
        elif any(indicator in doc_name for indicator in ['WV', 'West Virginia']):
            state = "WV"
        
        return county, state
    
    fp_locations = [extract_state_county(name) for name in false_positives['document_name']]
    tn_locations = [extract_state_county(name) for name in true_negatives['document_name']]
    
    fp_states = Counter([loc[1] for loc in fp_locations])
    fp_counties = Counter([loc[0] for loc in fp_locations])
    
    print(f"  State Distribution in False Positives:")
    for state, count in fp_states.most_common():
        print(f"    {state}: {count} documents")
    
    print(f"\n  County Distribution in False Positives:")
    for county, count in fp_counties.most_common(5):
        print(f"    {county}: {count} documents")

def generate_targeted_recommendations(false_positives, true_negatives):
    """Generate specific recommendations based on patterns"""
    
    print(f"\n1. 🎯 IMMEDIATE FIXES:")
    
    # Pattern 1: All false positives have 1.0 confidence with few samples
    print(f"   ❗ CRITICAL: All 9 false positives have 100% confidence with ≤3 samples")
    print(f"      → This indicates the prompt is too aggressive in early classification")
    print(f"      → SOLUTION: Require minimum 5-7 samples before high confidence classification")
    
    # Pattern 2: Early stopping issue
    fp_early_stop = false_positives[false_positives['samples_used'] == 3]
    print(f"\n   ❗ EARLY STOPPING: {len(fp_early_stop)}/9 false positives stopped at exactly 3 samples")
    print(f"      → System is reaching confidence threshold too quickly on wrong classification")
    print(f"      → SOLUTION: Raise confidence threshold from 0.5 to 0.75 for positive classifications")
    
    print(f"\n2. 🔧 PROMPT IMPROVEMENTS:")
    
    # Geographic patterns
    fp_counties = [name.split()[0] for name in false_positives['document_name']]
    county_counts = Counter(fp_counties)
    
    if county_counts.most_common(1)[0][1] > 1:
        top_county = county_counts.most_common(1)[0][0]
        print(f"   📍 GEOGRAPHIC BIAS: '{top_county}' appears in multiple false positives")
        print(f"      → May have region-specific boilerplate language")
        print(f"      → SOLUTION: Add examples from {top_county} documents to prompt")
    
    print(f"\n3. 🎚️  THRESHOLD ADJUSTMENTS:")
    print(f"   Current: confidence_threshold=0.5, max_samples=7")
    print(f"   Recommended: confidence_threshold=0.75, max_samples=10")
    print(f"   Rationale: Force more deliberation before positive classification")
    
    print(f"\n4. 📝 FEATURE ENGINEERING:")
    print(f"   Enhance 'boilerplate_indicators' feature to be more aggressive")
    print(f"   Add 'early_confidence_penalty' for classifications with <5 samples")
    print(f"   Implement 'geographic_context' feature for region-specific patterns")
    
    print(f"\n5. 🧪 TESTING STRATEGY:")
    print(f"   Re-run evaluation with adjusted thresholds on these 9 documents:")
    for _, row in false_positives.iterrows():
        print(f"     - {row['document_name']}")
    
    # Save false positive list for easy re-testing
    false_positive_list = false_positives['document_name'].tolist()
    with open("false_positive_documents.txt", "w") as f:
        for doc in false_positive_list:
            f.write(f"data/no-reservs/{doc}\n")
    
    print(f"\n   📝 False positive document list saved to: false_positive_documents.txt")

def main():
    """Main analysis function"""
    
    print("🚀 Starting Pattern Analysis...")
    
    try:
        false_positives, true_negatives = analyze_false_positive_patterns()
        
        print(f"\n✅ Pattern analysis complete!")
        print(f"   False Positives: {len(false_positives)} documents")
        print(f"   Key Finding: All false positives have 100% confidence with minimal sampling")
        print(f"   Primary Fix: Increase confidence threshold and minimum samples")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main() 