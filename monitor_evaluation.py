#!/usr/bin/env python3
"""
Simple monitoring script to check evaluation progress
"""

import time
import subprocess
import os
from pathlib import Path

def check_evaluation_status():
    """Check if evaluation is running and show progress"""
    
    print("🔍 EVALUATION MONITORING")
    print("=" * 50)
    
    # Check if process is running
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'evaluate_optimized_no_reservations.py' in result.stdout:
            print("✅ Evaluation is RUNNING")
            
            # Count lines in process output to estimate progress
            lines = result.stdout.split('\n')
            for line in lines:
                if 'evaluate_optimized_no_reservations.py' in line:
                    print(f"   Process: {line.split()[1]}")
                    break
        else:
            print("❌ Evaluation is NOT running")
    except Exception as e:
        print(f"Error checking process: {e}")
    
    # Check results directory
    results_dir = Path("optimized_evaluation_results")
    if results_dir.exists():
        files = list(results_dir.glob("*"))
        print(f"\n📁 Results directory: {len(files)} files")
        for file in files:
            print(f"   - {file.name}")
    else:
        print("\n📁 Results directory: Not created yet")
    
    # Check data directories
    no_reservs = len(list(Path("data/no-reservs").glob("*.pdf")))
    reservs = len(list(Path("data/reservs").glob("*.pdf")))
    total = no_reservs + reservs
    
    print(f"\n📊 Dataset: {total} documents total")
    print(f"   - No reservations: {no_reservs}")
    print(f"   - Has reservations: {reservs}")
    
    return True

if __name__ == "__main__":
    check_evaluation_status() 