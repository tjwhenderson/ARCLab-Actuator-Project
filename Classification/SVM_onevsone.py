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

df = df.dropna()
    

for index, row in df.iterrows():
    for i, classification in enumerate(classes):
        if row["Actuator Type"] == classification:
            lbl.append(i)


# drop first column now that we have labels 
df = df.drop('Actuator Type', axis=1)

# change dataframe to hold double values and stress/strain
df = df[['Strain (%)','Stress (MPa)']].apply(pd.to_numeric)

# normalize the data
df = df.apply(np.log)

scaler = StandardScaler()
# scales data by removing mean and scaling to the variance
scaler.fit(df)

# applies scaler to dataset 
x = scaler.transform(df)

# casts list of labels as numpy array
y = np.array(lbl)

# set up classifier
clf = SVC(kernel = 'rbf')

# trains SVM
clf.fit(x, y)
