import streamlit as st

from src.models.NaiveBayes import nb_param_selector
from src.models.NeuralNetwork import nn_param_selector
from src.models.RandomForet import rf_param_selector
from src.models.DecisionTree import dt_param_selector
from src.models.LogisticRegression import lr_param_selector
from src.models.KNearesNeighbors import knn_param_selector
from src.models.SVC import svc_param_selector
from src.models.GradientBoosting import gb_param_selector


def model_selector():
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

    return model_type, model