import numpy  as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
#Loadingdataset
dataset=pd.read_csv('Salary_Data.csv') 
#Feature Extraction
X=dataset.iloc[:,:-1].values 
y = dataset.iloc[:,-1].values 
#splitting Train & Test data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3) 
#Linear Rgression Model Creation
reg_model = LinearRegression() 
#ModelTraining(Fitthemodel) 
reg_model.fit(X_train,y_train) 
#Model Prediction
y_pred=reg_model.predict(X_test)
 #Finding R-Sqaure value
print("R-Sqaure value(accuracy):",r2_score(y_test,y_pred)) 
#Visualizing the graph
plt.scatter(X_test,y_test, color='red') 
plt.plot(X_test, y_pred,color='blue')
plt.show()
