# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Car_Price.csv")
df

df.head()

df.tail()

df.shape

df.info()

df.describe()

df.isnull().sum()

# Make all the strings in the same format
string_columns=list(df.dtypes[df.dtypes=='object'].index)
for col in string_columns:
    df[col]=df[col].str.lower().str.replace(' ','_')
    df[col]=df[col].str.lower().str.replace('-','_')

plt.figure(figsize=(20,20))
sns.heatmap(df.corr() , annot=True);

df.CarName.value_counts()

df.groupby('CarName').mean()['price']

df[df['horsepower']<=100]['price'].mean()

df[df['horsepower']>=100]['price'].mean()

df.groupby('curbweight')['price'].mean()

fueltype_le=LabelEncoder()
df['fueltype']= fueltype_le.fit_transform(df.fueltype)
enginelocation_le=LabelEncoder()
df['enginelocation']=enginelocation_le.fit_transform(df.enginelocation)
cylindernumber_le=LabelEncoder()
df['cylindernumber']=cylindernumber_le.fit_transform(df.cylindernumber)
enginetype_le=LabelEncoder()
df['enginetype']=enginetype_le.fit_transform(df.enginetype)
carbody_le=LabelEncoder()
df['carbody']=carbody_le.fit_transform(df.carbody)
aspiration_le=LabelEncoder()
df['aspiration']=aspiration_le.fit_transform(df.aspiration)

df.columns

X=df.drop(["car_ID","CarName","doornumber","drivewheel","enginelocation","fuelsystem","symboling",
           'compressionratio','peakrpm','citympg','highwaympg','carheight','stroke'],axis=1)
y=df['price']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3,shuffle=True,random_state = 8)

sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

model=LinearRegression()
model.fit(X_train, y_train)

y_pred= model.predict(X_test)

model.score(X_test,y_test)

plt.scatter(y_test, y_pred)
plt.xlabel('Actual prices')
plt.ylabel('Predicted prices')
plt.show()
