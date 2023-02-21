import streamlit as st
import pickle
import numpy as np
from scipy.special import inv_boxcox


# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

boxcox_param = 0.3854186331879921



st.title("Sales Predictor")

# Item
Item_Weight = st.selectbox('Item_Weight',np.arange(4.555000,21.350000,4.268117,dtype=int))
Item_Fat_Content = st.selectbox('Weight of the Laptop',df['Item_Fat_Content'].unique())
Item_Type = st.selectbox('Type of item',df['Item_Type'].unique())
Item_MRP = st.selectbox('MRP',np.arange(31.290000,266.888400,61.848723,dtype=int))

VisSqrt = st.selectbox('Visibility',np.arange(0.059789,0.471016,0.087842))


# Outlet
Outlet_Identifier = st.selectbox('Outlet_Identifier',df['Outlet_Identifier'].unique())
Outlet_Establishment_Year = st.selectbox('Outlet_Establishment_Year',df['Outlet_Establishment_Year'].unique())
Outlet_Size = st.selectbox('Outlet_Size',df['Outlet_Size'].unique())
Outlet_Location_Type = st.selectbox('Outlet_Location_Type',df['Outlet_Location_Type'].unique())
Outlet_Type = st.selectbox('Outlet_Type',df['Outlet_Type'].unique())



if st.button('Predict Price'):

    query = np.array([Item_Weight,Item_Fat_Content,Item_Type,Item_MRP,VisSqrt,
       Outlet_Identifier, Outlet_Establishment_Year,Outlet_Size,
       Outlet_Location_Type,Outlet_Type])
    query = query.reshape(1,10)
    pred=(pipe.predict(query)[0])
    pred=inv_boxcox(pred,0.3854186331879921)

    st.title(f'query: {query}')
    st.title(f"The predicted price of this configuration is {pred}")


