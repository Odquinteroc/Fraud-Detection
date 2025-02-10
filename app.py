import streamlit as st
import pandas as pd
import pickle

st.title("This app is to predict weather a transaction is Fraud or not")


model_knn=pickle.load(open('knn_model.pkl','rb'))
model_dt=pickle.load(open('dt_model.pkl','rb'))
model_lr=pickle.load(open('logistic_r_model.pkl','rb'))


#load the dataset
df =pd.read_csv("creditcard.csv")
y= df['Class']
X=df.drop('Class', axis=1)

# user dat
user_input={}

for col in X.columns:
    user_input[col]=st.slider(col,X[col].min(),X[col].max())

df1=pd.DataFrame(user_input,index=[0])
df2 = df1[['V3','V4','V7','V10','V11','V12','V14','V16','V17']].copy(deep = True)

models={"Logistic Regression":model_lr,"KNN":model_knn,"Decision Tree":model_dt}

selected_model=st.selectbox("Select a model",("Logistic Regression","KNN","Decision Tree"))

if st.button("Predict"):
    prediction=models[selected_model].predict(df2)[0]
    if prediction == 1:
        st.write(f'This transaction is: Fraud')
    if prediction == 0:
        st.write(f'This transaction is: No Fraud')
    
    
