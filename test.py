# Test the pandas as pd, and how to use it to read a CSV file and display its contents. The code imports the pandas library, reads a CSV file located at 'C:\\Jet machine learning\\Aircraft Engine Dataset\\FD002\\RUL_targets.csv', and prints the contents of the DataFrame created from the CSV file.
import random

from matplotlib.pylab import seed
import pandas as pd
df = pd.read_csv('C:\\Jet machine learning\\Aircraft Engine Dataset\\FD002\\RUL_targets.csv')
print(pd.DataFrame(df)) #print the table in one go

#test numpy as np
import numpy as np
np.random.seed(42) #print the random seed of numpy
Random_array = np.random.rand(5) #create a random array of 5 elements
print(f"Random array: {np.round(Random_array,2)}")#print the random array
random_number = np.random.rand() #create a random number
print(f"Random number: {random_number:.2f}") #print the random number