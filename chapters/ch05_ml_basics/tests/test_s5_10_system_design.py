import csv
from chapters.ch05_ml_basics.api import init_csv_logger, log_row

def test_csv_logger_and_append(tmp_path):
    path = tmp_path / "log.csv"
    fields = ["epoch", "train_loss", "val_loss"]

    # init creates header
    init_csv_logger(str(path), fields, overwrite=True)
    assert path.exists()

    # append two rows
    log_row(str(path), {"epoch": 1, "train_loss": 0.9, "val_loss": 1.1}, fieldnames=fields)
    log_row(str(path), {"epoch": 2, "train_loss": 0.7, "val_loss": 0.9}, fieldnames=fields)

    rows = list(csv.DictReader(open(path, newline="")))
    assert len(rows) == 2
    assert rows[0]["epoch"] == "1" and rows[1]["epoch"] == "2"

def test_logger_writes_header_if_missing(tmp_path):
    path = tmp_path / "metrics.csv"
    # no init_csv_logger call; first log_row should create header automatically
    row = {"epoch": 0, "acc": 0.5}
    log_row(str(path), row)
    rows = list(csv.DictReader(open(path, newline="")))
    assert rows and set(rows[0].keys()) == set(row.keys())
