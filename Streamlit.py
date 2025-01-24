import math
import pandas as pd
import pandas
import numpy as np
import streamlit as st
# Information
st.sidebar.markdown("## Authors' information")
st.sidebar.markdown("Authors: Van-Thanh Pham and Jong-Sung Kim")
st.sidebar.caption("Department of Quantum and Nuclear Engineering, Sejong University, Korea")
st.sidebar.caption("Emails: phamthanhwru@gmail.com and kimjsbat@sejong.ac.kr")

#Tên bài báo
st.title ("Hybrid machine learning with optimization algorithm and resampling method for the swelling rate prediction of irradiated 316 stainless steels") 

# Chèn sơ đồ nghiên cứu
# st.header("1. Layout of this study")
# check1=st.checkbox('1.1 Display layout of this investigation')
# if check1:
#    st.image("Fig. 1.jpg", caption="Layout of this study")
# # Hiển thị dữ liệu
# st.header("2. Dataset")
# check2=st.checkbox('2.1 Display dataset')
# if check2:
#    Database="Dataset.csv"
#    df = pandas.read_csv(Database)
#    df.head()
#    st.write(df)
# st.header("3. Modeling approach")
# check3=st.checkbox('3.1 Display structure of Random Forest model')
# if check3:
#    st.image("Fig2.jpg", caption="Overview on structure of Random Forest model") 
	
#Make a prediction
st.header("Predicting the swelling rate of irradiated 316 stainless steels")
st.subheader("Input variables")
col1, col2, col3, col4, col5 =st.columns(5)

with col1:
   X1=st.slider("Fe (wt. %)", 62.70, 71.21)
   X2 = st.slider("Cr (wt. %)", 14.86, 18.97)
   X3= st.slider("Mn (wt. %)", 0.01, 3.89)
   X4 = st.slider("Si (wt. %)", 0.01, 1.96)
   X5 = st.slider("Co (wt. %)", 0.00, 4.45)
	
with col2:	

   X6= st.slider("Mo (wt. %)", 0.01, 4.93)
   X7 = st.slider("Ni (wt. %)", 9.96, 14.04)
   X8=st.slider("C (wt. %)", 0.00, 0.13)	
   X9 = st.slider("N (wt. %)", 0.00, 0.13)
   X10= st.slider("B /1000 (wt. %)", 0.00, 5.70)

with col3:		
   X11 = st.slider("P /100 (wt. %)", 0.10, 4.00)
   X12 = st.slider("S /100 (wt. %)", 0.40, 1.90)
   X13= st.slider("Al (wt. %)", 0.00, 0.10)
   X14 = st.slider("Ti /100 (wt. %)", 0.00, 2.00)   
   X15=st.slider("Nb /100 (wt. %)", 0.00, 1.00)	
	
with col4:	
   X16 = st.slider("Ta /100 (wt. %)", 0.00, 2.00)
   X17= st.slider("Pb /100 (wt. %)", 0.00, 2.00)
   X18 = st.slider("Cu (wt. %)", 0.00, 4.47)
   X19 = st.slider("Pre-irr. flue. x10^22(n/cm2)", 0.00, 3.86)
	
with col5:		
   X20= st.slider("Irr. flue. x10^22(n/cm2)", 1.53, 14.00)	
   X21 = st.slider("Temperature (C)", 376.00, 780.10)  
   X22=st.slider("Stress (MPa)", 0.0, 403.0)
   X23 = st.slider("Disl. Dens. x10^14(1/m2)", 1.50, 30.00)



from sklearn.model_selection import train_test_split, KFold
#from sklearn.ensemble import GradientBoostingRegressor
#import catboost as cb
from catboost import CatBoostRegressor

#Model

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


df = pd.read_csv('Swelling_data_316SS_Thanh_WERCS.csv')
# X_ori = df[['D','W','den','fc','E','L','C','P']].values
X_ori = df[['Fe','Cr','Mn','Si','Co','Mo','Ni','C','N','B','P','S','Al','Ti','Nb','Ta','Pb','Cu','Pre-irr','Irr','Temp','Stre','Dis']].values
y = df['Swell'].values

X = scaler.fit_transform(X_ori)	
X_train=X
y_train=y

n_estimators = 502
learning_rate = 0.223
depth = 4
l2_leaf_reg = 7
        
cat_clf_n = CatBoostRegressor(n_estimators =n_estimators,learning_rate = learning_rate, depth=depth,l2_leaf_reg=l2_leaf_reg, random_state=42)
cat_clf_n.fit(X_train, y_train)


Inputdata = [X1, X2, X3, X4, X5, X6, X7, X8, X9, X10/1000, X11/100, X12/100, X13, X14/100, X15/100, X16/100, X17/100,X18, X19, X20, X21, X22, X23*100000000000000]


from numpy import asarray
Newdata1 = asarray([Inputdata])
print(Newdata1)
Newdata=scaler.transform(Newdata1)

fc_pred2 = cat_clf_n.predict(Newdata)

st.subheader("Output variable")
if st.button("Predict"):
    import streamlit as st
    import time
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
       time.sleep(0.01)
       my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()
    st.success(f"Your predicted swelling rate (%) obtained from WERCS-FHO-CGB model is: {(fc_pred2)}")

