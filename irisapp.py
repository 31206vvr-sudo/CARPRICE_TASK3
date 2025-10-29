import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load and preprocess data
df = pd.read_csv('Iris.csv')
df = df.drop('Id', axis=1)
df['Species'] = df['Species'].astype('category').cat.codes

X = df.drop('Species', axis=1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

st.title('Iris Flower Classification')
st.sidebar.header('Input Features')
sepal_length = st.sidebar.slider('Sepal Length (cm)', float(df['SepalLengthCm'].min()), float(df['SepalLengthCm'].max()), float(df['SepalLengthCm'].mean()))
sepal_width = st.sidebar.slider('Sepal Width (cm)', float(df['SepalWidthCm'].min()), float(df['SepalWidthCm'].max()), float(df['SepalWidthCm'].mean()))
petal_length = st.sidebar.slider('Petal Length (cm)', float(df['PetalLengthCm'].min()), float(df['PetalLengthCm'].max()), float(df['PetalLengthCm'].mean()))
petal_width = st.sidebar.slider('Petal Width (cm)', float(df['PetalWidthCm'].min()), float(df['PetalWidthCm'].max()), float(df['PetalWidthCm'].mean()))

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]

species_label = df['Species'].astype('category').cat.categories[prediction]
st.subheader('Predicted Species')
st.write(species_label)
