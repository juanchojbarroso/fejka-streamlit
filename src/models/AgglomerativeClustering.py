import streamlit as st
from sklearn.cluster import AgglomerativeClustering


def agglomerative_param_selector():
    n_clusters = st.number_input("n_clusters", 2, 10, 2, 1)
    linkage = st.selectbox("linkage", ["ward", "complete", "average"])
    affinity = st.selectbox(
        "affinity", ["euclidean", "l1", "l2", "manhattan", "cosine"])
    model = AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkage, affinity=affinity)
    return model
