import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import pickle

data = pd.read_csv("dataset_37_diabetes.csv")

classes={'tested_positive':1,'tested_negative':0}
data['class']=[classes[item] for item in data['class']]
Y=data['class']
X=data.drop(columns=['class'])
# Y.value_counts()

from imblearn.combine import SMOTETomek
smk=SMOTETomek(random_state=42)
x_res,y_res=smk.fit_sample(X,Y)

df=x_res
df['result']=y_res
# df.head()

df = df.rename(columns = {'preg': 'num_preg', 'plas': 'gluc_conc',
                           'pres':'diastolic_bp','skin':'skin_thick',
                           'insu':'insulin','mass':'bmi',
                          'pedi':'Diab_Pred'})
# df.head()

x=df.drop(columns=['result']).values
y=df['result'].values

"""
print('The mean of Glucose Concentration : {0}'.format(df['gluc_conc'].mean()))
print('The mean of Diastolic BP : {0}'.format(df['diastolic_bp'].mean()))
print('The mean of Skin Thickness : {0}'.format(df['skin_thick'].mean()))
print('The mean of Insulin : {0}'.format(df['insulin'].mean()))
print('The mean of BMI : {0}'.format(df['bmi'].mean()))
print('The mean of Diab Pedigree : {0}'.format(df['Diab_Pred'].mean()))
print('The mean of Age : {0}'.format(df['age'].mean()))

O/P:
The mean of Glucose Concentration : 125.11659663865547
The mean of Diastolic BP : 69.24264705882354
The mean of Skin Thickness : 21.077731092436974
The mean of Insulin : 85.24159663865547
The mean of BMI : 32.82792384447866
The mean of Diab Pedigree : 0.5025532575363812
The mean of Age : 33.8235294117647
"""


# Replacing 0 with mean values except pregnancies column
df['gluc_conc']=df['gluc_conc'].replace(0,125.12)
df['diastolic_bp']=df['diastolic_bp'].replace(0,69.24)
df['skin_thick']=df['skin_thick'].replace(0,21.1)
df['insulin']=df['insulin'].replace(0,85.24)
df['bmi']=df['bmi'].replace(0,32.83)
df['Diab_Pred']=df['Diab_Pred'].replace(0,0.503)
df['age']=df['age'].replace(0,33)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)

# print("total number of rows : {0}".format(len(df)))
# print("number of rows missing glucose_conc: {0}".format(len(df.loc[df['gluc_conc'] == 0])))
# print("number of rows missing diastolic_bp: {0}".format(len(df.loc[df['diastolic_bp'] == 0])))
# print("number of rows missing skin_thick: {0}".format(len(df.loc[df['skin_thick'] == 0])))
# print("number of rows missing insulin: {0}".format(len(df.loc[df['insulin'] == 0])))
# print("number of rows missing bmi: {0}".format(len(df.loc[df['bmi'] == 0])))
# print("number of rows missing diab_pred: {0}".format(len(df.loc[df['Diab_Pred'] == 0])))
# print("number of rows missing age: {0}".format(len(df.loc[df['age'] == 0])))


from sklearn.ensemble import RandomForestClassifier
random_forest_model=RandomForestClassifier(random_state=10)
random_forest_model.fit(x_train,y_train.ravel())
prediction=random_forest_model.predict(x_test)

# Accuracy of model 
# from sklearn import metrics 
# print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, prediction)))


"""
Hyper parameter optimization

params={
    'criterion':['gini','entropy'],
    'max_depth':range(1,10),
    'min_samples_split':range(1,14),
    'min_samples_leaf':range(1,5)
    
}

gridi=GridSearchCV(random_forest_model,param_grid=params,cv=10,verbose=1,n_jobs=-1)

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

from datetime import datetime
start_time = timer(None) 
gridi.fit(x_train,y_train.ravel())
timer(start_time) 

gridi.best_params_

    {'criterion': 'gini',
    'max_depth': 9,
    'min_samples_leaf': 1,
    'min_samples_split': 2}
gridi.best_estimator_
"""

# pred=[6,98,72,35,0,33.6,0.627,20]
# random_forest_model.predict([pred])[0]

file = open("model.pkl","wb")
pickle.dump(random_forest_model,file)
file.close()