import streamlit as st
import pandas as pd

st.title('🤖 Machine learning App')

st.write('This app builds a machine learning model!')

with st.expander('Data'):
  st.write('**Raw data**')
  df= pd.read_csv('https://raw.githubusercontent.com/geetikavasistha-01/gv-machinelearning/refs/heads/master/penguins.csv')
  df


  st.write('**X**')
  X= df.drop('species'. axis=1)
  X

  st.write('**y**')
  y=df.species
  y
