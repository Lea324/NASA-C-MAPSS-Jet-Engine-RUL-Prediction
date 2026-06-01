import os
main_csv_local_path = 'RUL_targets.csv'
for dirname, _, filenames in os.walk(r'C:\Jet machine learning'):
    for filename in filenames:
        if filename == main_csv_local_path:
            DATA_DIR = os.path.join(dirname, filename)
            print(DATA_DIR)

import numpy as np #NumPy (numerical computing)
import pandas as pd #Pandas (data processing)
import matplotlib.pyplot as plt #Matplotlib (graphs and visualization)
import scipy #SciPy (engineering calculations)
import seaborn as sns #Seaborn (statistical data visualization)
import warnings #Warnings (to manage warning messages)