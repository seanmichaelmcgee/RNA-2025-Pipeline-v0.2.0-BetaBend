#!/usr/bin/env python3
"""
Test script to verify the RNA IDs parameter works correctly.
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Test RNA IDs parameter")
    parser.add_argument("--rna-ids", type=str, nargs='+', 
                        help="Specific RNA IDs to validate", default=None)
    args = parser.parse_args()
    
    if args.rna_ids:
        print(f"RNA IDs received: {', '.join(args.rna_ids)}")
    else:
        print("No RNA IDs provided")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())