import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc


#importing dataset
data = pd.read_csv("datasets_13996_18858_WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(data)


#DATA CLEANING
#dropping customer id column because there is a different value for every
#person so it will not help in churn prediction

data.drop(['customerID'], axis = 1, inplace = True)
print(data)
#convert column TotalCharges to numeric

print(data["TotalCharges"].dtype)
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"].str.split(" ").str.get(0))
print(data["TotalCharges"].dtype)

#Replacing null values with mean column value

data["TotalCharges"] = data["TotalCharges"].fillna(value = np.mean(data["TotalCharges"]))

#Replacing 'No internet service' with 'No' for every column value
# because they mean the same thing and 'No phone service' with 'No'
for col in data:
    if data[col].dtypes == 'object':
        print(f'{col} : {data[col].unique()}')

data.replace(['No internet service', 'No phone service'], ['No', 'No'], inplace = True)

#Converting 'Yes' to 1s and 'No' to 0s for every column
yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']

for col in yes_no_columns:
    data[col].replace({'Yes': 1, 'No': 0}, inplace = True)

data['gender'].replace({'Female': 1, 'Male': 0}, inplace = True)

#One hot-encoding other categorical related columns

encoder = LabelBinarizer()
InternetService_enc = encoder.fit_transform(data['InternetService'])
data['InternetService'] = InternetService_enc

encoder = LabelBinarizer()
Contract_enc = encoder.fit_transform(data['Contract'])
data['Contract'] = Contract_enc

encoder = LabelBinarizer()
PaymentMethod_enc = encoder.fit_transform(data['PaymentMethod'])
data['PaymentMethod'] = PaymentMethod_enc

cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']


#scaling data
scaler = MinMaxScaler()
data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale] )



#calculating correlation
corr_matrix = data.corr()

print(corr_matrix)



#features which have the strongest corrrelation with churning
print(corr_matrix['Churn'].sort_values(ascending= False))

attributes = ["Contract", "MonthlyCharges", "PaperlessBilling","SeniorCitizen"]
scatter_matrix(data[attributes], figsize=(12, 8))
plt.show()

X = data[attributes]
#print(X)
y= data['Churn']
#print(y)

#Splitting data

X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
print("\n")
print(len(X_train), "train +", len(X_test), "test")

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


#Traing the model

model = XGBClassifier(use_label_encoder=False )

model.fit(X_train, Y_train)

y_pred= model.predict(X_test)

predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(Y_test, predictions )

print(accuracy)

#calculating AUC


xgb_fpr, xgb_tpr, threshold = roc_curve(Y_test, predictions)
auc_xgb = auc(xgb_fpr, xgb_tpr)

plt.figure(figsize=(5,5), dpi=100)
plt.plot(xgb_fpr, xgb_tpr, marker='.', label='Model AUC (auc= %0.3f)' % auc_xgb )

plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')

print("Area under curve is " + str(auc_xgb))
plt.legend()
plt.show()


pickle.dump(model, open('model.pkl', 'wb'))