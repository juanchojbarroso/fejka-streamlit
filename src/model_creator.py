
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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


def train_model(pipeline, X_train, y_train, X_test, y_test):
    start_time = time.time()

    pipeline.fit(X_train, y_train)

    train_pipeline_predict = pipeline.predict(X_train)
    test_pipeline_predict = pipeline.predict(X_test)

    train_accuracy = round(accuracy_score(
        y_train, train_pipeline_predict), 3)
    train_precision = round(precision_score(
        y_train, train_pipeline_predict, average="weighted"), 3)
    train_recall = round(recall_score(
        y_train, train_pipeline_predict, average="weighted"), 3)
    train_f1 = round(
        f1_score(y_train, train_pipeline_predict, average="weighted"), 3)

    test_accuracy = round(accuracy_score(y_test, test_pipeline_predict), 3)
    test_precision = round(precision_score(
        y_test, test_pipeline_predict, average="weighted"), 3)
    test_recall = round(recall_score(
        y_test, test_pipeline_predict, average="weighted"), 3)
    test_f1 = round(
        f1_score(y_test, test_pipeline_predict, average="weighted"), 3)

    duration = time.time() - start_time

    return pipeline, train_accuracy, train_precision, train_recall, train_f1, test_accuracy, test_precision, test_recall, test_f1, duration


def create_model(X, y, model_type, model, test_size, random_state):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    num_features, cat_features = get_columns_types(
        X_train)

    preprocessor = get_preprocessor(num_features, cat_features)

    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline, train_accuracy, train_precision, train_recall, train_f1, test_accuracy, test_precision, test_recall, test_f1, duration = train_model(
        model_pipeline, X_train, y_train, X_test, y_test)

    st.warning(f"Training took {duration:.3f} seconds")

    col1, col2 = st.beta_columns(2)

    # Show train results
    col1.markdown("**Train Accuracy**: " +
                  str(train_accuracy))
    col1.markdown("**Train Precision**: " + str(train_precision))
    col1.markdown("**Train Recall**: " + str(train_recall))
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
    col2.markdown("**Test Precision**: " + str(test_precision))
    col2.markdown("**Test Recall**: " + str(test_recall))
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
