import os
import pandas as pd

note_dir = "./note"
files = [
    "discharge.csv.gz",
    "discharge_detail.csv.gz",
    "radiology.csv.gz",
    "radiology_detail.csv.gz"
]

for file_name in files:
    file_path = os.path.join(note_dir, file_name)
    print(f"\n--- Checking: {file_name} ---")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, compression="gzip")
        print(f"Columns found: {list(df.columns)}")
        print(f"Total rows: {len(df)}")
        print("Sample row:")
        print(df.head(1))
    else:
        print(f"ERROR: File {file_name} not found!")