import streamlit as st
from src.layout import views

st.set_page_config(
    page_title="Fejka", layout="wide"
)

# navigation links
link = st.sidebar.radio(label='Links', options=['Charts', 'Machine Learning'])
views(link)

