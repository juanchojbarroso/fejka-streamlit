import streamlit as st

from src.models.NaiveBayes import nb_param_selector
from src.models.NeuralNetwork import nn_param_selector
from src.models.RandomForet import rf_param_selector
from src.models.DecisionTree import dt_param_selector
from src.models.LogisticRegression import lr_param_selector
from src.models.KNearesNeighbors import knn_param_selector
from src.models.SVC import svc_param_selector
from src.models.GradientBoosting import gb_param_selector
from src.models.KMeans import kmeans_param_selector
from src.models.DBSCAN import dbscan_param_selector
from src.models.AgglomerativeClustering import agglomerative_param_selector


def model_selector_supervised():
    model_training_container = st.beta_expander("Train a model", True)
    with model_training_container:
        model_type = st.selectbox(
            "Choose a model",
            (
                "Decision Tree",
                "Random Forest",
                "Logistic Regression",
                "Gradient Boosting",
                "Neural Network",
                "K Nearest Neighbors",
                "Gaussian Naive Bayes",
                "SVC",
                "KMeans",
                "DBSCAN",
                "Agglomerative Clustering",
            ),
        )

        if model_type == "Logistic Regression":
            model = lr_param_selector()

        elif model_type == "Decision Tree":
            model = dt_param_selector()

        elif model_type == "Random Forest":
            model = rf_param_selector()

        elif model_type == "Neural Network":
            model = nn_param_selector()

        elif model_type == "K Nearest Neighbors":
            model = knn_param_selector()

        elif model_type == "Gaussian Naive Bayes":
            model = nb_param_selector()

        elif model_type == "SVC":
            model = svc_param_selector()

        elif model_type == "Gradient Boosting":
            model = gb_param_selector()

        elif model_type == "KMeans":
            model = kmeans_param_selector()

        elif model_type == "DBSCAN":
            model = dbscan_param_selector()

        elif model_type == "Agglomerative Clustering":
            model = agglomerative_param_selector()

    return model_type, model


def model_selector_unsupervised():
    model_training_container = st.beta_expander("Train a model", True)
    with model_training_container:
        model_type = st.selectbox(
            "Choose a model",
            (
                'K Means Clustering',
                'Hierarchical Clustering'
            ),
        )

        if model_type == "K Means Clustering":
            model = kmeans_param_selector()

        elif model_type == "Hierarchical Clustering":
            model = agglomerative_param_selector()

    return model_type, model
