#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder
import joblib
le = LabelEncoder()

#import shap
#import matplotlib.pyplot as plt 
#from sklearn.ensemble import RandomForestRegressor



hide_menu = """
<style>
#MainMenu {
    visibility:hidden;
}
footer{
    visibility:hidden;
}
</style>
"""
st.markdown(hide_menu,unsafe_allow_html=True)


df=pd.read_csv('cleareddata.csv')
y=df['PRICE']
X=df[['ADDRESS','LEVEL','CONSTRUCTION_YEAR','BATHROOMS','BEDROOMS','SIZES']]
X['ADDRESS_TRANSFORMED'] = le.fit_transform(df['ADDRESS'])

ADDRESS_DIC=X[['ADDRESS','ADDRESS_TRANSFORMED']]
ADDRESS_DIC = ADDRESS_DIC.drop_duplicates().reset_index()

X = X.drop('ADDRESS', 1)





@st.cache(allow_output_mutation=True)
def datasetmodel(X,y):

    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    model.fit(X_train,y_train)
    # model prediction

    y_pred = model.predict(X_test)
    
    # Save the trained model as a pickle string.
    saved_model = pickle.dumps(model)

    # Load the pickled model
    model_from_pickle1 = pickle.loads(saved_model)
    return(model_from_pickle1)
    




model=datasetmodel(X,y)




st.title('Greek Real Estate Data')
st.sidebar.write('Characteristics')


#st.dataframe(df)
address=df.ADDRESS.unique()
address.sort()
area=st.sidebar.selectbox("Choose Area",(address))
st.header(area)
dfarea=df.loc[df['ADDRESS'] == area]

bedrooms=df.BEDROOMS.unique()
bedrooms.sort()
BEDROOMS = st.sidebar.slider('Number of bedrooms',min_value=int(min(bedrooms)),max_value=int(max(bedrooms)))
st.write('Number of Bedrooms : ',BEDROOMS)

bathrooms=df.BEDROOMS.unique()
bathrooms.sort()
BATHROOMS = st.sidebar.slider('Number of bathrooms',min_value=int(min(bathrooms)),max_value=int(max(bathrooms)))
st.write('Number of Bathrooms : ',BATHROOMS)

size=df.SIZES.unique()
size.sort()
SIZE = st.sidebar.slider('Size in square meters',min_value=int('20'),max_value=int('500'))
st.write('Size in Square Meters: ', SIZE)

level=df.LEVEL.unique()
level.sort()
LEVELS = st.sidebar.slider('Level',min_value=int(min(level)),max_value=int(max(level)))
st.write('Level : ',LEVELS)

year=df.CONSTRUCTION_YEAR.unique()
year.sort()
YEAR = st.sidebar.slider('Year of Construction',min_value=int('1958'),max_value=int(max(year)))
st.write('Construction Year : ',YEAR)

mean_price=dfarea.PRICE.mean()
st.write('AVERAGE PRICE IN ',area,' : '+ str(int(mean_price))+'€')
st.write('Number of appartments for sale: ',len(dfarea.ID))






c=0
while c < len(ADDRESS_DIC.ADDRESS):
    if ADDRESS_DIC['ADDRESS'][c]==area:
        area_trans=int(ADDRESS_DIC['ADDRESS_TRANSFORMED'][c])
        c=len(ADDRESS_DIC.ADDRESS)
    
    c=c+1





inp = np.array([LEVELS,YEAR,BATHROOMS,BEDROOMS,SIZE,area_trans])
inp=inp.reshape((1,-1))





mcprice=int(model.predict(inp))
st.subheader('The Predicted Price is: '+str(mcprice)+'€')

