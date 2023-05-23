import streamlit as st
from sklearn.cluster import DBSCAN


def dbscan_param_selector():
    eps = st.slider("eps", 0.1, 1.0, 0.5, 0.1)
    min_samples = st.number_input("min_samples", 1, 10, 5, 1)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model