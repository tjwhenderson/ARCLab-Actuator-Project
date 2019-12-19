### Actuator Classification

Gaussian_Mix.m 
Input: actuator name, e.g. "SMA"
Output: mean and variance.

Train_SVM.m
Input: completed data matrix, list of labels.
Output: trained SVM model Mdl.

testSVM.m
Input: trained SVM model Mdl, actuator features wanted, e.g [0.1,0.1,0.1]
Output: Type of actuator to be used, probabilities of using other types of actuators.
