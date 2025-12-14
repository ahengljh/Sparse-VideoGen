#!/usr/bin/env python3
"""
Generate all figures for the CTCA/CTAA paper.

Usage:
    cd paper/figures
    python generate_all.py
"""

import subprocess
import sys

scripts = [
    'generate_overview.py',
    'generate_ctca_workflow.py',
    'generate_ctaa_tiers.py',
    'generate_temporal_coherence.py',
    'generate_speedup_analysis.py',
]

def main():
    print("=" * 60)
    print("Generating all paper figures...")
    print("=" * 60)

    for script in scripts:
        print(f"\nRunning {script}...")
        try:
            result = subprocess.run([sys.executable, script], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✓ {script} completed successfully")
                if result.stdout:
                    print(f"    {result.stdout.strip()}")
            else:
                print(f"  ✗ {script} failed")
                if result.stderr:
                    print(f"    Error: {result.stderr.strip()}")
        except Exception as e:
            print(f"  ✗ Error running {script}: {e}")

    print("\n" + "=" * 60)
    print("Figure generation complete!")
    print("=" * 60)

    # List generated files
    import os
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    png_files = [f for f in os.listdir('.') if f.endswith('.png')]

    print(f"\nGenerated PDF files: {len(pdf_files)}")
    for f in sorted(pdf_files):
        print(f"  - {f}")

    print(f"\nGenerated PNG files: {len(png_files)}")
    for f in sorted(png_files):
        print(f"  - {f}")

if __name__ == "__main__":
    main()
