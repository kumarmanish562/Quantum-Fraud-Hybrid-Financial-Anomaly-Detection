import argparse
import sys
import os

# Ensure the script can see ml_engine if running from different pwd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_engine.trainers.train_classical import run_classical_training
from ml_engine.trainers.train_hybrid import run_hybrid_training

def main():
    parser = argparse.ArgumentParser(description="Quantum Fraud Detection CLI")
    parser.add_argument('mode', choices=['train-classical', 'train-hybrid', 'all'], help="Mode to run")
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    
    if args.mode == 'train-classical' or args.mode == 'all':
        print("=== Starting Classical Training ===")
        run_classical_training()
        
    if args.mode == 'train-hybrid' or args.mode == 'all':
        print("\n=== Starting Hybrid Quantum Training ===")
        run_hybrid_training()

if __name__ == "__main__":
    main()
