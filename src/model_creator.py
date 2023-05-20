
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

# def train_model(pipeline, x_train, y_train, x_test, y_test):
#     t0 = time.time()
#     pipeline.fit(x_train, y_train)
#     duration = time.time() - t0
#     y_train_pred = pipeline.predict(x_train)
#     y_test_pred = pipeline.predict(x_test)

#     train_accuracy = np.round(accuracy_score(y_train, y_train_pred), 3)
#     train_f1 = np.round(f1_score(y_train, y_train_pred, average="weighted"), 3)

#     test_accuracy = np.round(accuracy_score(y_test, y_test_pred), 3)
#     test_f1 = np.round(f1_score(y_test, y_test_pred, average="weighted"), 3)

#     return pipeline, train_accuracy, train_f1, test_accuracy, test_f1, duration


def create_model(X, y, model_type, model, test_size, random_state):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

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