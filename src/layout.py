import streamlit as st
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay


from src.graph_controls import graph_controls
from src.utility import load_dataframe

from src.models.NaiveBayes import nb_param_selector
from src.models.NeuralNetwork import nn_param_selector
from src.models.RandomForet import rf_param_selector
from src.models.DecisionTree import dt_param_selector
from src.models.LogisticRegression import lr_param_selector
from src.models.KNearesNeighbors import knn_param_selector
from src.models.SVC import svc_param_selector
from src.models.GradientBoosting import gb_param_selector

from sklearn.linear_model import LogisticRegression


import time
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


def get_preprocessor(num_features, cat_features):
    numeric_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )
    categorical_tranformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_tranformer, cat_features)
        ]
    )

    return preprocessor


def train_model(pipeline, x_train, y_train, x_test, y_test):
    t0 = time.time()
    pipeline.fit(x_train, y_train)
    duration = time.time() - t0
    y_train_pred = pipeline.predict(x_train)
    y_test_pred = pipeline.predict(x_test)

    train_accuracy = np.round(accuracy_score(y_train, y_train_pred), 3)
    train_f1 = np.round(f1_score(y_train, y_train_pred, average="weighted"), 3)

    test_accuracy = np.round(accuracy_score(y_test, y_test_pred), 3)
    test_f1 = np.round(f1_score(y_test, y_test_pred, average="weighted"), 3)

    return pipeline, train_accuracy, train_f1, test_accuracy, test_f1, duration


def get_columns_types(df):
    numeric_features = []
    categorical_features = []

    numerics = df._get_numeric_data().columns

    for column in numerics:
        numeric_features.append(column)

    categoricals = list(set(df.columns) - set(numerics))

    for column in categoricals:
        categorical_features.append(column)

    return numeric_features, categorical_features


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


def views(link):
    """
    Helper function for directing users to various pages.
    :param link: str, option selected from the radio button
    :return:
    """
    if link == 'Charts':
        st.header("Charts")
        st.sidebar.subheader('Settings')
        st.sidebar.subheader("Upload your data")
        clean_data = st.sidebar.checkbox(label='Clean data')
        uploaded_file = st.sidebar.file_uploader(label="Upload your csv or excel file here.",
                                                 accept_multiple_files=False,
                                                 type=['csv', 'xlsx'])

        if uploaded_file is not None:
            df, columns = load_dataframe(
                uploaded_file=uploaded_file, clean_data=clean_data)

            st.sidebar.subheader("Visualize your data")

            try:
                st.subheader("Data view")
                number_of_rows = st.sidebar.number_input(
                    label='Select number of rows', min_value=2)

                st.dataframe(df.head(number_of_rows))
            except Exception as e:
                print(e)

            st.sidebar.subheader("Theme selection")

            theme_selection = st.sidebar.selectbox(label="Select your themes",
                                                   options=['plotly', 'plotly_white',
                                                            'ggplot2',
                                                            'seaborn', 'simple_white'])
            st.sidebar.subheader("Chart selection")
            chart_type = st.sidebar.selectbox(label="Select your chart type.",
                                              options=['Scatter plots', 'Density contour',
                                                       'Sunburst', 'Pie Charts', 'Density heatmaps',
                                                       'Histogram', 'Box plots', 'Tree maps',
                                                       'Violin plots', ])  # 'Line plots',

            selected_columns = st.multiselect("Select features", columns)

            graph_controls(chart_type=chart_type, df=df,
                           dropdown_options=selected_columns, template=theme_selection)

    if link == 'Machine Learning':
        st.header('Machine Learning')
        st.sidebar.subheader('Settings')
        st.sidebar.subheader("Upload your data")
        clean_data = st.sidebar.checkbox(label='Clean data')
        uploaded_file = st.sidebar.file_uploader(label="Upload your csv or excel file here.",
                                                 accept_multiple_files=False,
                                                 type=['csv', 'xlsx'])

        if uploaded_file is not None:
            df, columns = load_dataframe(
                uploaded_file=uploaded_file, clean_data=clean_data)

            st.sidebar.subheader("Visualize your data")

            try:
                st.subheader("Data view")
                number_of_rows = st.sidebar.number_input(
                    label='Select number of rows', min_value=2)

                st.dataframe(df.head(number_of_rows))
            except Exception as e:
                print(e)

            selected_columns = st.multiselect("Select features", columns)
            data = df[selected_columns]
            target_options = data.columns
            chosen_target = st.selectbox("Select target", (target_options))

            X = data.loc[:, data.columns != chosen_target]
            X.columns = data.loc[:, data.columns != chosen_target].columns
            y = data[chosen_target]

            default_value_random_state = 42
            random_state = st.slider(
                'Random State', min_value=1, max_value=200, value=default_value_random_state)
            test_size = st.selectbox("Select test data size", (0.2, 0.3))

            st.dataframe(df.head(number_of_rows))

            if len(X) > 1:
                model_type, model = model_selector()
                model_btn = st.button('CREATE MODEL')

                if model_btn:

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state)

                    st.dataframe(df.head(number_of_rows))

                    num_features, cat_features = get_columns_types(
                        X_train)

                    preprocessor = get_preprocessor(num_features, cat_features)

                    pipeline = Pipeline(steps=[
                        ("preprocessor", preprocessor),
                        ("model", model)
                    ])

                    start_time = time.time()

                    pipeline.fit(X_train, y_train)

                    train_accuracy = round(pipeline.score(X_train, y_train), 3)
                    train_f1 = round(
                        f1_score(y_train, pipeline.predict(X_train), average="weighted"), 3)

                    test_accuracy = round(pipeline.score(X_test, y_test), 3)
                    test_f1 = round(
                        f1_score(y_test, pipeline.predict(X_test), average="weighted"), 3)

                    duration = time.time() - start_time

                    st.warning(f"Training took {duration:.3f} seconds")

                    col1, col2 = st.beta_columns(2)

                    # Show train results
                    col1.markdown("**Train Accuracy**: " +
                                  str(train_accuracy))
                    col1.markdown("**Tranin F1 Score**: " + str(train_f1))

                    display = ConfusionMatrixDisplay.from_estimator(
                        pipeline,
                        X_train,
                        y_train,
                        cmap=plt.cm.Blues,
                        normalize=None,
                    )
                    display.ax_.set_title(
                        "Confusion Matrix for " + model_type + " (Train)")
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    col1.pyplot()

                    # Show test results
                    col2.markdown("**Test Accuracy**: " + str(test_accuracy))
                    col2.markdown("**Test F1 Score**: " + str(test_f1))

                    display = ConfusionMatrixDisplay.from_estimator(
                        pipeline,
                        X_test,
                        y_test,
                        cmap=plt.cm.Reds,
                        normalize=None,
                    )
                    display.ax_.set_title(
                        "Confusion Matrix for " + model_type + " (Test)")
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    col2.pyplot()
