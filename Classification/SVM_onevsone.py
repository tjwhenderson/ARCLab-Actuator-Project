import pandas as pd 
import numpy as np
from matrix_completion import *
import sklearn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

# drop all features except stress, strain
df = df.drop(['Bandwidth (Hz)', 'Efficiency (%)', 'Power Density (W/g)'], axis=1)


# drop missing data points 
df = df.dropna()

# append each actuator type to the lbl array
for index, row in df.iterrows():
    for i, classification in enumerate(classes):
        if row["Actuator Type"] == classification:
            lbl.append(i)


# drop first column now that we have labels 
df = df.drop('Actuator Type', axis=1)

# change dataframe to hold double values and stress/strain
df = df[['Strain (%)','Stress (MPa)']].apply(pd.to_numeric)
df = df.apply(pd.to_numeric)

# check if there are any missing data points
#df_imputed = df
#df.isnull().sum(axis = 0)

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

# normalize data
df_imputed = df_imputed.apply(np.log)

# remove non-numeric values in stress/strain columns
df_imputed['Strain (%)'] = pd.to_numeric(df_imputed['Strain (%)'], errors='coerce')
df_imputed['Stress (MPa)'] = pd.to_numeric(df_imputed['Stress (MPa)'], errors='coerce')

# dataframe of stress and strain properties
df_imputed = df_imputed.dropna()

# change df to numpy array
df_np = df_imputed.to_numpy()

scaler = StandardScaler()
# scales data by removing mean and scaling to the variance
scaler.fit(df_np)

# applies scaler to dataset 
x = scaler.transform(df_np)

# casts list of labels as numpy array
y = np.array(lbl)

# set up classifier
clf = SVC(kernel = 'rbf', C=1000)

# trains SVM
clf.fit(x, y)



y_predict = clf.predict(x)

sklearn.metrics.accuracy_score(y_predict, y)


# create a copy of df_imputed for plotting
df_plt = df_imputed

# create actuator type to add column to df (for plot labeling)
actuator_type = []

# iterate through labels and find corresponding actuator type
for i, act_type in enumerate(lbl):
    for j, classif in enumerate(classes):
        if act_type == j: 
            actuator_type.append(classes[j])

# add a column to df_plt for actuator type 
df_plt['Actuator Type'] = actuator_type


#x_lin = np.linspace(min(df_imputed['Strain (%)']), max(df_imputed['Strain (%)']), np.floor(max(df_imputed['Strain (%)'])).astype(np.int))
#y_lin = np.linspace(min(df_imputed['Stress (MPa)']), max(df_imputed['Stress (MPa)']),np.floor(max(df_imputed['Stress (MPa)'])).astype(np.int))
#X, Y = np.meshgrid(x_lin,y_lin)


# from plot multi-class SGD example...

h = .02  # step size in the mesh

# create a mesh to plot in
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# set figure size to be bigger
figure(figsize=(12, 8), dpi=80)


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.axis('tight')

# set color palette for classes
# colors = "rybcwm"
ys = [i+x+(i*x)**2 for i in range(10)]
colors = cm.Spectral(np.linspace(0, 1, len(ys)))


# Plot also the training points

for i, color in zip(clf.classes_, colors):
    idx = np.where(y == i)
    plt.scatter(x[idx, 0], x[idx, 1], c=color, label=classes[i],
                cmap=plt.cm.Spectral, edgecolor='black', s=20)
plt.title("Actuator Properties")
plt.axis('tight')
plt.xlabel('Strain (%)')
plt.ylabel('Stress (MPa)')

plt.legend()
plt.show()


# some classes barely show up so only 5 classes are predicted??
np.unique(y_predict)
 

# split data into train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2)


# set up classifier
clf = SVC(kernel = 'rbf', C=1000, probability=True)

# train training data set 
clf.fit(x_train, y_train)


y_test_predict = clf.predict(x_test)
y_test_prob = clf.predict_proba(x_test)
y_train_predict = clf.predict(x_train)
y_train_prob = clf.predict_proba(x_train)

sklearn.metrics.accuracy_score(y_test_predict, y_test)

sklearn.metrics.accuracy_score(y_train_predict, y_train)


print(classification_report(y_test, y_test_predict))


#both preds and truths are same shape m by n (m is number of predictions and n is number of classes)
def top_n_accuracy(preds, truths, n):
    best_n = np.argsort(preds, axis=1)[:,-n:]
    ts = truths
    successes = 0
    for i in range(ts.shape[0]):
      if ts[i] in best_n[i,:]:
        successes += 1
    return float(successes)/ts.shape[0]


# top-3 accuracy
top_n_accuracy(y_test_prob, y_test, 3)
top_n_accuracy(y_train_prob, y_train, 3)




