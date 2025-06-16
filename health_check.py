#!/usr/bin/env python3
"""
Health check script for Render deployment
"""

import os
import sys

def check_environment():
    """Check if all required environment variables are set"""
    print("🔍 Checking environment...")
    
    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set!")
        return False
    
    print(f"✅ ANTHROPIC_API_KEY found (length: {len(api_key)})")
    return True

def check_imports():
    """Check if all required packages can be imported"""
    print("📦 Checking imports...")
    
    try:
        import anthropic
        print(f"✅ anthropic {anthropic.__version__}")
    except ImportError as e:
        print(f"❌ Failed to import anthropic: {e}")
        return False
    
    try:
        import flask
        print(f"✅ flask {flask.__version__}")
    except ImportError as e:
        print(f"❌ Failed to import flask: {e}")
        return False
    
    try:
        import fitz
        print("✅ PyMuPDF imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PyMuPDF: {e}")
        return False
    
    return True

def check_anthropic_client():
    """Test Anthropic client initialization"""
    print("🤖 Testing Anthropic client...")
    
    try:
        import anthropic
        api_key = os.getenv('ANTHROPIC_API_KEY')
        
        # Try basic initialization
        client = anthropic.Anthropic(api_key=api_key)
        print("✅ Anthropic client initialized successfully")
        
        # Try a simple API call
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello"}]
        )
        print("✅ Anthropic API call successful")
        return True
        
    except Exception as e:
        print(f"❌ Anthropic client test failed: {e}")
        return False

def main():
    """Run all health checks"""
    print("🏥 Render Deployment Health Check")
    print("=" * 40)
    
    checks = [
        ("Environment", check_environment),
        ("Imports", check_imports),
        ("Anthropic Client", check_anthropic_client)
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n{name}:")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All health checks passed!")
        sys.exit(0)
    else:
        print("💥 Some health checks failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 