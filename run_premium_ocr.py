#!/usr/bin/env python3
"""
Premium OCR Evaluation Runner
============================

Quick runner script for premium OCR evaluation with optimized configurations.
Focus: Maximum accuracy with state-of-the-art methods only.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_api_keys():
    """Check if API keys are available"""
    has_openai = bool(os.getenv('OPENAI_API_KEY'))
    has_anthropic = bool(os.getenv('ANTHROPIC_API_KEY'))
    
    return {
        'openai': has_openai,
        'anthropic': has_anthropic,
        'any_api': has_openai or has_anthropic
    }

def run_premium_evaluation(config_name: str):
    """Run premium OCR evaluation with predefined configurations"""
    
    api_status = check_api_keys()
    
    configs = {
        'quick': {
            'engines': ['trocr-printed', 'donut'],
            'max_docs': 2,
            'description': 'Quick test with best open-source models',
            'requires_api': False
        },
        'transformers': {
            'engines': ['trocr-printed', 'trocr-handwritten', 'donut'],
            'max_docs': 3,
            'description': 'Comprehensive transformer-based OCR evaluation',
            'requires_api': False
        },
        'llm-only': {
            'engines': ['gpt4o', 'claude'],
            'max_docs': 2,
            'description': 'Premium LLM vision models (requires API keys)',
            'requires_api': True
        },
        'premium': {
            'engines': ['trocr-printed', 'donut', 'gpt4o'],
            'max_docs': 3,
            'description': 'Best overall: TrOCR + Donut + GPT-4o',
            'requires_api': True
        },
        'complete': {
            'engines': ['trocr-printed', 'trocr-handwritten', 'donut', 'gpt4o', 'claude'],
            'max_docs': 4,
            'description': 'Complete premium evaluation (all SOTA methods)',
            'requires_api': True
        },
        'research': {
            'engines': ['trocr-printed', 'trocr-handwritten', 'donut', 'surya', 'gpt4o', 'claude'],
            'max_docs': 5,
            'description': 'Research-grade evaluation with experimental models',
            'requires_api': True
        }
    }
    
    if config_name not in configs:
        print(f"❌ Unknown configuration: {config_name}")
        print(f"📋 Available configurations: {list(configs.keys())}")
        return
    
    config = configs[config_name]
    
    # Check API requirements
    if config['requires_api'] and not api_status['any_api']:
        print(f"❌ Configuration '{config_name}' requires API keys")
        print("💡 Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY environment variables")
        print("🔄 Try 'quick' or 'transformers' for API-free evaluation")
        return
    
    # Filter engines based on available APIs
    available_engines = []
    for engine in config['engines']:
        if engine == 'gpt4o' and not api_status['openai']:
            print(f"⚠️  Skipping {engine} - no OpenAI API key")
            continue
        elif engine == 'claude' and not api_status['anthropic']:
            print(f"⚠️  Skipping {engine} - no Anthropic API key")
            continue
        else:
            available_engines.append(engine)
    
    if not available_engines:
        print("❌ No engines available for this configuration")
        return
    
    print(f"🏆 Running {config_name} evaluation: {config['description']}")
    print(f"🔧 Engines: {', '.join(available_engines)}")
    print(f"📄 Max documents: {config['max_docs']}")
    
    # Build command
    cmd = [
        sys.executable, 'src/ocr_evaluation.py',
        '--engines'] + available_engines + [
        '--max_docs', str(config['max_docs']),
        '--output_dir', f'data/outputs/premium_ocr_{config_name}'
    ]
    
    print(f"📋 Command: {' '.join(cmd)}")
    print("🚀 Starting evaluation...\n")
    
    # Run evaluation
    try:
        subprocess.run(cmd, check=True)
        print(f"\n🎉 {config_name} evaluation completed successfully!")
        print(f"📁 Results saved to: data/outputs/premium_ocr_{config_name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed: {e}")
    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")

def show_status():
    """Show system status and available configurations"""
    print("🏆 Premium OCR Evaluation System Status")
    print("=" * 50)
    
    # Check API keys
    api_status = check_api_keys()
    print("🔑 API Keys:")
    print(f"   OpenAI (GPT-4o): {'✅ Available' if api_status['openai'] else '❌ Missing'}")
    print(f"   Anthropic (Claude): {'✅ Available' if api_status['anthropic'] else '❌ Missing'}")
    
    # Check data
    data_dirs = ['data/reservs', 'data/no-reservs', 'data/samples']
    print("\n📄 Sample Data:")
    for data_dir in data_dirs:
        path = Path(data_dir)
        if path.exists():
            pdf_count = len(list(path.glob("*.pdf")))
            print(f"   {data_dir}: {pdf_count} PDFs")
        else:
            print(f"   {data_dir}: ❌ Not found")
    
    print("\n🎯 Recommended Configurations:")
    if not api_status['any_api']:
        print("   🔸 quick        - Fast test (no API needed)")
        print("   🔸 transformers - Comprehensive (no API needed)")
    else:
        print("   🔸 premium      - Best overall performance")
        print("   🔸 complete     - Full evaluation")
    
    print("\n💡 To get started:")
    print("   python run_premium_ocr.py quick")

def main():
    if len(sys.argv) == 1:
        show_status()
        return
    
    if len(sys.argv) != 2:
        print("Usage: python run_premium_ocr.py <config>")
        print("\n🏆 Premium OCR Configurations:")
        print("  quick        - Fast test with TrOCR + Donut (no API)")
        print("  transformers - All transformer models (no API)")
        print("  llm-only     - GPT-4o + Claude only (requires API)")
        print("  premium      - Best overall: TrOCR + Donut + GPT-4o")
        print("  complete     - All premium methods")
        print("  research     - Experimental + all methods")
        print("\n💡 Examples:")
        print("  python run_premium_ocr.py quick")
        print("  python run_premium_ocr.py premium")
        print("\n📊 Status:")
        print("  python run_premium_ocr.py")
        return
    
    config_name = sys.argv[1]
    
    if config_name in ['status', 'info', 'check']:
        show_status()
        return
    
    run_premium_evaluation(config_name)

if __name__ == "__main__":
    main() 