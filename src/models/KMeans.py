import streamlit as st
from sklearn.cluster import KMeans


def kmeans_param_selector():
    n_clusters = st.number_input(
        "Choose the number of clusters", 1, 8, step=1, key='noofclusters')

    default_value_random_state = 42
    random_state = st.slider(
        'Random State', min_value=1, max_value=200, value=default_value_random_state)

    init = st.selectbox("init", ["k-means++", "random"])

    max_iter = st.number_input("max_iter", 100, 1000, 500, 100)

    model = KMeans(n_clusters=n_clusters, random_state=random_state,
                   init=init, max_iter=max_iter, n_init=10)

    return model
