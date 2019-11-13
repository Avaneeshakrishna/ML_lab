# Program 5 - Naive Bayesian Classifier (With reduced lines of code ) 

import math
import statistics as st
from sklearn.model_selection import train_test_split
import pandas as pd 

def summarizeByClass(x_tr,y_tr):    
    separated = {}  # Create a dictionary with  labels as keys 1 and 0  
    for i in range(len(x_train)):
        x, y = x_tr[i],y_tr[i]
        if (y not in separated):
            separated[y] = []
        separated[y].append(x)    
    
    summary = {} # to store mean and std of +ve and -ve instances
    for lbl, subset in separated.items():    
        summary[lbl] = [ (st.mean(attribute), st.stdev(attribute)) 
                         for attribute in zip(*subset)];  #zip(*res) transposes a matrix (2-d array/list)

    return summary
 
#For continuous attributes p is estimated using Gaussion distribution
def estimateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
 
def predict(summaries, testVector):
    bestLabel, bestProb = None, -1
    p = {}
    for lbl, mean_std in summaries.items():
        #class and attribute information as mean and sd
        p[lbl] = 1
        for i in range(len(mean_std)):
            mean, stdev = mean_std[i] 
            x = testVector[i]
            p[lbl] *= estimateProbability(x, mean, stdev);
            
        #assigns that class which has he highest prob
        if bestLabel is None or p[lbl] > bestProb:
            bestProb = p[lbl]
            bestLabel = lbl
            
    return bestLabel
 
def do_classification_compute_accuracy(summaries, test_x, test_y):
    correct = 0
    for i in range(len(test_x)):
        result = predict(summaries, test_x[i])
        if result == test_y[i]:
            correct = correct + 1

    accuracy = (correct/float(len(test_x))) * 100.0
    return accuracy

# Main program 
df=pd.read_csv('data51.csv',header=None)
cols = [0,1,2,3,4,5,6,7]
df_x = df[df.columns[cols]]
df_y = df[df.columns[8]]

X = df_x.values.tolist()
Y = df_y.values.tolist()

x_train, x_test, y_train, y_test = train_test_split(X,Y)

print('Dataset loaded...')
print('Total instances available :',len(X))
print('Total attributes present  :',len(X[0])-1)
print("First Five instances of dataset:")
for i in range(5):
    print(i+1 , ':' , X[i])
        
print('\nDataset is split into training and testing set.')
print('Training examples = {0} \nTesting examples  = {1}'.format
      (len(x_train), len(x_test)))

summaries = summarizeByClass(x_train,y_train);
accuracy = do_classification_compute_accuracy(summaries,x_test,y_test)

print('\nAccuracy of the Naive Baysian Classifier is :', accuracy)