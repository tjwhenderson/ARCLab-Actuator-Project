import pandas as pd 
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# read google spreadsheet data saved as csv
df = pd.read_csv("original_data.csv")

# define actuator class names
classes = ["PZT", "DEA", "IPMC", "SMA", "SCP", "SFA", "TSA", "EAP"]

# delete first column from dataframe
df = df.drop('Reference', axis=1)

# define new first column as labels for actuator type
actuator_type = df['Actuator Type']

# extract actuator type and set as label
lbl = [] 

# remove SMP actuator types
df = df[df['Actuator Type'] != 'SMP']
    

for index, row in df.iterrows():
    for i, classification in enumerate(classes):
        if row["Actuator Type"] == classification:
            lbl.append(i)

print(lbl[1:10])

# drop first column now that we have labels 
df = df.drop('Actuator Type', axis=1)

# change dataframe to hold double values and stress/strain
df = df[['Strain (%)','Stress (MPa)']].apply(pd.to_numeric)

# normalize the data
df = df.apply(np.log)

from matrix_completion import *

# casts everything to float and numpy matrix
df_numeric = df.to_numpy()

shape = df_numeric.shape


mask = np.ones((shape[0], shape[1]))

# set missing values to 0
mask[np.isnan(df_numeric)] = 0

imputed = nuclear_norm_solve(df_numeric, mask)    
    
    
# transforming numpy array to dataframe and setting columsn and indicies
df_imputed = pd.DataFrame(imputed)
df_imputed.columns = df.columns
df_imputed.index = df.index

scaler = StandardScaler()
# scales data by removing mean and scaling to the variance
scaler.fit(df_imputed)

# applies scaler to dataset 
x = scaler.transform(df_imputed)

# casts list of labels as numpy array
y = np.array(lbl)

# set up classifier
clf = SVC(kernel = 'rbf')

# trains SVM
clf.fit(x, y)

# predict on training data set
y_predict = clf.predict(x)

# get accuracy metrics
print(sklearn.metrics.accuracy_score(y_predict, y))
