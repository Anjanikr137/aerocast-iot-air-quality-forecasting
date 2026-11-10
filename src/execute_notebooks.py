"""
Programmatic execution of all 6 Jupyter Notebooks to verify end-to-end execution.
"""

import os
import sys
import nbformat
from nbclient import NotebookClient

NOTEBOOKS = [
    "01_Dataset_Audit_EDA.ipynb",
    "02_Feature_Engineering.ipynb",
    "03_Linear_Regression.ipynb",
    "04_Random_Forest.ipynb",
    "05_XGBoost.ipynb",
    "06_Artificial_Neural_Network.ipynb"
]

def execute_all():
    nb_dir = "notebooks"
    print(f"Starting execution of all {len(NOTEBOOKS)} notebooks...")
    
    for nb_file in NOTEBOOKS:
        nb_path = os.path.join(nb_dir, nb_file)
        print(f"\n==========================================")
        print(f"Executing: {nb_file}...")
        print(f"==========================================")
        
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
            
        client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={'metadata': {'path': nb_dir}})
        
        try:
            client.execute()
            with open(nb_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)
            print(f"[SUCCESS] {nb_file} executed and saved successfully with complete outputs!")
        except Exception as e:
            print(f"[ERROR] Failed executing {nb_file}: {e}")
            raise e

    print("\nAll 6 notebooks successfully executed from beginning to end!")

if __name__ == '__main__':
    execute_all()

