import streamlit as st

st.set_page_config(page_icon="📟", 
                   page_title="ML Studio",
                   layout="wide")




st.title("ML Studio")
st.subheader("Machine Learning Studio")
st.write("""This is a simple machine learning studio to train and test models.""")
st.write("You can upload your dataset, select a model, and train it.")
st.write("You can also test the model and see the results.")
st.page_link("pages/Dataset_Upload.py", label="Upload Dataset",icon="📤")