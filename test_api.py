#!/usr/bin/env python3
"""
Simple test script to verify Anthropic API connectivity
"""

import os
from document_classifier import DocumentProcessor

def test_api_connection():
    """Test if the API key and connection are working"""
    
    print("🧪 Testing Anthropic API Connection...")
    print("=" * 50)
    
    # Check if API key is set
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY environment variable not set!")
        print("Please set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return False
    
    print(f"✅ API key found (length: {len(api_key)} characters)")
    
    try:
        # Initialize processor
        print("🔧 Initializing document processor...")
        processor = DocumentProcessor()
        print("✅ Document processor initialized successfully!")
        
        # Test simple classification
        print("🧪 Testing simple text classification...")
        test_text = "This is a simple deed with no mineral rights reservations."
        
        result = processor.classifier.classify_document(
            test_text, 
            max_samples=2,  # Just 2 samples for quick test
            confidence_threshold=0.6
        )
        
        print(f"✅ Classification test successful!")
        print(f"   Result: {result.predicted_class}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Samples used: {result.samples_used}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_api_connection()
    if success:
        print("\n🎉 All tests passed! Your API connection is working.")
    else:
        print("\n💥 Tests failed. Please check your API key and internet connection.") 