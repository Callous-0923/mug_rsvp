# -*- coding: utf-8 -*-
import csv
import os
import time
from typing import Dict, Any

class CSVLogger:
    def __init__(self, csv_path: str, headers=None):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.csv_path = csv_path
        self.headers = headers
        self._file = open(csv_path, 'w', newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)
        if headers:
            self._writer.writerow(headers)
            self._file.flush()

    def log(self, row):
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()

class SimpleTimer:
    def __init__(self): self.t0 = None
    def start(self): self.t0 = time.time()
    def stop(self):
        if self.t0 is None: return 0.0
        return time.time() - self.t0
