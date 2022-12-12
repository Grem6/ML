
## Importing the required library

import re 
import nltk ##not used
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import warnings
import time
warnings.filterwarnings('ignore')
# from csv import reader, writer ## for writing output data

df=pd.read_csv(r'C:\Users\unnik\Downloads\assignment\COVIDSenti-A.csv') 
# df=pd.read_csv(r'/COVIDSenti-A.csv') ## change to relative path to re-run the code (repo)

## doing the necessary preprocessing steps


def data_details():
    """
    Generates dataset details.
    
    Outputs: [dataset.columns, dataset.all().isnull(), dataset.describe(), null_checker.sum]
    
    Parameters: None
    
    """
    
    print(f'data details : \n {df.columns}')
    print('\n') 

    null_checker = df.all().isnull()
    
    if not (null_checker['tweet'] & null_checker['label']): 
        ##checks for null values in dataset, skips if none
        print(f'Data info : \n {df.describe()}')
    else:
        print(f'{null_checker.sum()}')
        
    print('\n')     
        
    print(f'Column details : \n {df.label.value_counts()}')
    print('\n')
    print(f'Unique columns : \n {df.label.unique()}')
    print('\n')
    
data_details()    
def pattern_matcher(user_text, pattern): 
    ## function to match patterns in data to remove.
    """
    Pattern Matcher with re library.

    Args:
        user_text (string): receives the pandas series as a string input.
        pattern (string): user defined pattern to match

    Returns:
        string : user_text
    """
    r = re.findall(pattern, user_text)
    for i in r:
        user_text = re.sub(i,'',user_text)
    return user_text

## data vectorizing steps to match and remove necessary strings from datastet.

df['tweet'] = np.vectorize(pattern_matcher)((df['tweet']), '@[\w]*')
df['tweet'] = df['tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
df['tweet'] = df['tweet'].str.replace('[^a-zA-Z#]+',' ')
df['tweet'] = df['tweet'].str.replace('#',' ')
df['tweet']=  df['tweet'].str.lower()
df['tweet'] = df['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))

print(f'Cleaned data : {df.head()}')
print('\n')
sample = df[df['label']=='neg'].loc[2,'tweet']
print(f'Cleaned sample : {sample}')


## tokenized data

tokenized_output = df['tweet'].apply(lambda x: x.split())
lemmatizer = WordNetLemmatizer()
tokenized_output = tokenized_output.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
print(tokenized_output)

for i in range(len(tokenized_output)):
    tokenized_output[i] = ' '.join(tokenized_output[i])
    
df['tweets']  = tokenized_output
print(df['tweets'])
df.drop('tweet',axis=1,inplace=True)
print('\n')  
print(f'Tokenized data : {df.head()}')
print('\n')  

## splitting into dependent and independent variables
## performing the rest of the steps to finalize the model

 
X = df['tweets']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state = 101)
vectorized = TfidfVectorizer(max_features = 5000)
vectorized.fit(df['tweets'])
X_train_vectorized = vectorized.transform(X_train)
X_test_vectorized = vectorized.transform(X_test)

print(X_train_vectorized)
print("\n")
print(X_test_vectorized)
print("\n")

##fitting the svm model to the vectorized data

model = SVC(probability = True, kernel = 'linear')
model.fit(X_train_vectorized, y_train )

predictions = model.predict(X_test_vectorized)
Predicted_data = pd.DataFrame()
Predicted_data['tweets'] = X_test
Predicted_data['label'] = predictions
print(f'New data : {Predicted_data}')
print("\n")

## unique values in the predicted data

unique  = Predicted_data['label'].value_counts()
print(f'New unique value : {unique}')
print("\n")

## final accuracy of the model

accuracy = accuracy_score(predictions, y_test)*100
print(f'Model accuracy = {accuracy}')
print("\n")

## classification report generated

final_result  = classification_report(y_test, predictions)
print(f'Classification Report : {classification_report(y_test, predictions)}') 
##check result txt for program output

