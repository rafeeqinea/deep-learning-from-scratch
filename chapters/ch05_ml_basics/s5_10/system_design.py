import csv
import os
from typing import Dict, List, Optional

def init_csv_logger(path: str, fieldnames: List[str], overwrite: bool = False) -> None:
    """
    Create CSV file with header if needed. If overwrite=True, truncates file.
    """
    exists = os.path.exists(path)
    mode = "w" if (overwrite or not exists) else "a"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if overwrite or not exists:
            writer.writeheader()

def log_row(path: str, row: Dict[str, object], fieldnames: Optional[List[str]] = None) -> None:
    """
    Append a row to CSV, writing header if file absent.
    """
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        if fieldnames is None:
            fieldnames = list(row.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
