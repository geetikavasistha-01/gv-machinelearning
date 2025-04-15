import streamlit as st
import pandas as pd
import numpy as np
st.title('🤖 Machine learning App')

st.write('This app builds a machine learning model!')

with st.expander('Data'):
  st.write('**Raw data**')
  df= pd.read_csv('https://raw.githubusercontent.com/geetikavasistha-01/gv-machinelearning/refs/heads/master/penguins.csv')
  df


  st.write('**X**')
  X= df.drop('species', axis=1)
  X

  st.write('**y**')
  y=df.species
  y

with st.expander('Data Visualization'): 
  st.scatter_chart(data=df,x='bill_length_mm' , y='body_mass_g',color='species')
