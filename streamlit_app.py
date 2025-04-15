import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Title and intro
st.title('🐧 Penguin Species Predictor')
st.info('This app builds a machine learning model to predict penguin species!')

# Load and display data
with st.expander('Data'):
    st.write('**Raw data**')
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    st.dataframe(df)

    st.write('**X (Features)**')
    X_raw = df.drop('species', axis=1)
    st.dataframe(X_raw)

    st.write('**y (Target)**')
    y_raw = df.species
    st.dataframe(y_raw)

# Data visualization
with st.expander('Data visualization'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Sidebar input
with st.sidebar:
    st.header('📥 Input Features')
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
    gender = st.selectbox('Sex', ('male', 'female'))

    # Create DataFrame for input
    data = {'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': gender}
    input_df = pd.DataFrame(data, index=[0])

# Combine input with full data for encoding consistency
input_penguins = pd.concat([input_df, X_raw], axis=0)

# Show input data
with st.expander('Input features'):
    st.write('**Input Penguin**')
    st.dataframe(input_df)

    st.write('**Combined Data (for encoding)**')
    st.dataframe(input_penguins)

# Encode categorical features
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, columns=encode)

# Ensure same columns as training set
X = df_penguins[1:]
input_row = df_penguins[:1]

# Encode target variable
target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
y = y_raw.apply(lambda val: target_mapper[val])

# Display prepared data
with st.expander('Data preparation'):
    st.write('**Encoded Input (X)**')
    st.dataframe(input_row)

    st.write('**Encoded y**')
    st.dataframe(y)

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

# Make prediction
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

# Format prediction probabilities
df_prediction_proba = pd.DataFrame(prediction_proba, columns=['Adelie', 'Chinstrap', 'Gentoo'])

# Display prediction probabilities
st.subheader('🔍 Prediction Probability')
st.dataframe(df_prediction_proba,
             column_config={
                 'Adelie': st.column_config.ProgressColumn('Adelie', format='%f', min_value=0, max_value=1),
                 'Chinstrap': st.column_config.ProgressColumn('Chinstrap', format='%f', min_value=0, max_value=1),
                 'Gentoo': st.column_config.ProgressColumn('Gentoo', format='%f', min_value=0, max_value=1),
             }, hide_index=True)

# Final predicted species
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(f"🎯 Predicted Penguin Species: **{penguins_species[prediction][0]}**")

