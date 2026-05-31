import os
main_csv_local_path = 'RUL_targets.csv'
for dirname, _, filenames in os.walk(r'C:\Jet machine learning'):
    for filename in filenames:
        if filename == main_csv_local_path:
            DATA_DIR = os.path.join(dirname, filename)
            print(DATA_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
