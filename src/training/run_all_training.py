"""
Execution runner for all 4 AeroCast model families.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.training.train_models import (
    load_processed_data,
    train_linear_regression,
    train_random_forest,
    train_xgboost,
    train_ann
)
from src.evaluation.consolidate_comparison import generate_consolidated_reports

def main():
    print("Loading preprocessed training, validation, and test datasets...")
    datasets, preprocessing = load_processed_data()
    
    print("\nPhase 5: Linear Regression Training...")
    train_linear_regression(datasets, preprocessing)
    
    print("\nPhase 6: Random Forest Training...")
    train_random_forest(datasets, preprocessing)
    
    print("\nPhase 7: XGBoost Training...")
    train_xgboost(datasets, preprocessing)
    
    print("\nPhase 8: ANN Training...")
    train_ann(datasets, preprocessing)
    
    print("\nPhase 9 & 10: Consolidated Comparison, Reporting & Verification...")
    generate_consolidated_reports()
    print("\nAll training, persistence, and reporting phases completed successfully!")

if __name__ == '__main__':
    main()

