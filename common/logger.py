# -*- coding: utf-8 -*-
import csv
import os

class Logger():
    def __init__(self, log_dir):
        try:
            os.makedirs(log_dir)
        except:
            raise OSError("The current directory already existed:)")
            
        filename = "training_progress.csv"
        file_dir = os.path.join(log_dir, filename)
        
        self.f = open(file_dir, "w", newline="")
        self.writer = csv.writer(self.f)
        
        self.log_count = 0

    def write(self, data):
        assert isinstance(data, list), "The data must be a list:)"
        
        self.writer.writerow(data)
        
    def close(self):
        self.f.close()