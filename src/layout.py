import streamlit as st
from src.file import upload_file
from src.ui_model_selector import model_selector_supervised, model_selector_unsupervised
from src.supervised_model_creator import create_model as create_model_supervised
# from src.unsupervised_model_creator import create_model as create_model_unsupervised

from src.graph_controls import graph_controls


from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, mean_squared_error
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import pandas as pd



tooltip_text = "Supervised models use labeled data and learn to predict known labels. Unsupervised models do not require labels and aim to discover hidden patterns in the data. Supervised models are evaluated by comparing predictions with the true labels. Unsupervised models are evaluated using specific metrics for each algorithm."


def views(link):
    """
    Helper function for directing users to various pages.
    :param link: str, option selected from the radio button
    :return:
    """
    if link == 'Charts':
        st.header("Charts")
        df, columns = upload_file()
        if columns is not None:
            st.subheader("Theme selection")
            theme_selection = st.selectbox(label="Select your themes",
                                           options=['plotly', 'plotly_white',
                                                    'ggplot2',
                                                    'seaborn', 'simple_white'])
            st.subheader("Chart selection")
            chart_type = st.selectbox(label="Select your chart type.",
                                      options=['Scatter plots', 'Density contour',
                                               'Sunburst', 'Pie Charts', 'Density heatmaps',
                                               'Histogram', 'Box plots', 'Tree maps',
                                               'Violin plots', ])  # 'Line plots',

            graph_controls(chart_type=chart_type, df=df,
                           dropdown_options=columns, template=theme_selection)
        else:
            st.subheader("Upload your data to create a chart")

    if link == 'Machine Learning':
        st.header('Machine Learning')
        df, columns = upload_file()

        if columns is not None:
            st.subheader("Learning model type")
            learning_model_type = st.radio(label='Select:', options=[
                                           'Supervised', 'Unsupervised'], help=tooltip_text)

            if learning_model_type == 'Supervised':
                selected_columns = st.multiselect("Select features", columns)
                data = df[selected_columns]
                target_options = data.columns
                chosen_target = st.selectbox(
                    "Select target", (target_options.insert(0, '<select>')), 0)
                if chosen_target != '<select>':
                    X = data.loc[:, data.columns != chosen_target]
                    X.columns = data.loc[:, data.columns !=
                                         chosen_target].columns
                    y = data[chosen_target]

                    default_value_random_state = 42
                    random_state = st.slider(
                        'Random State', min_value=1, max_value=200, value=default_value_random_state)
                    test_size = st.selectbox(
                        "Select test data size", (0.2, 0.3))

                    st.dataframe(df.head(3))

                    if len(X) > 1:
                        model_type, model = model_selector_supervised()
                        model_btn = st.button('CREATE MODEL')

                        if model_btn:
                            create_model_supervised(X, y, model_type, model, test_size,
                                                    random_state)
                else:
                    st.warning(
                        "Please select a target variable to create a model.")
            if learning_model_type == 'Unsupervised':
                selected_columns = st.multiselect("Select features", columns)
                X = df[selected_columns].values

                if len(selected_columns) < 1:
                    st.warning(
                        "Please select at least one feature to create a model.")
                else:
                    model_type, model = model_selector_unsupervised()

                    if model_type == 'K Means Clustering':

                        st.info(
                            "Use Elbow method to find the optimal nnumber of clusters")
                        if st.button("Elbow Method"):
                            wcss = []
                            fig, ax = plt.subplots()
                            for i in range(1, 11):
                                kmeans = KMeans(n_clusters=i, init='k-means++',
                                                max_iter=300, n_init=10)
                                kmeans.fit(X)
                                wcss.append(kmeans.inertia_)
                            ax.plot(range(1, 11), wcss)
                            ax.set_title('The Elbow Method')
                            ax.set_xlabel('Numbers of Clusters')
                            ax.set_ylabel('wcss')
                            st.pyplot(fig)

                        if st.button("CREATE CLUSTER", key='cluster'):
                            st.subheader("K means Clustering Results")
                            y_kmeans = model.fit_predict(X)
                            centroid = 'kmeans'
                            n_clusters = model.n_clusters
                            plot_values(n_clusters, y_kmeans, X,
                                        centroid,  model, model_type)

                    else:
                        st.info(
                            "Use Dendrogram to find the optimal number of clusters")

                        if st.button('Dendrogram'):
                            dendrogram = sch.dendrogram(
                                sch.linkage(X, method='ward'))
                            plt.title('Dendrogram')
                            plt.xlabel('Customers')
                            plt.ylabel('distance (euclidean')
                            st.pyplot()

                        if st.button("CREATE CLUSTER", key='cluster'):
                            st.subheader("Hierarchical Clustering Results")
                            y_kmeans = model.fit_predict(X)
                            centroid = 'hierarchy'
                            n_clusters = model.n_clusters
                            plot_values(n_clusters, y_kmeans,
                                        X, centroid, model, model_type)

        else:
            st.subheader("Upload your data to create a model")


def plot_values(n_clusters, y_kmeans, X, centroid, model, model_type):
    colors = ['red', 'blue', 'green', 'cyan', 'magenta',
              'sienna', 'lightpink', 'black', 'chocolate', 'violet']
    fig, ax = plt.subplots()
    for i in range(n_clusters):
        ax.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1],
                   s=100, c=colors[i], label='Cluster '+str(i+1))
    if centroid == 'kmeans':
        ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
                   s=200, c='yellow', label='Centroids')
    ax.set_title('Clusters based on:  ' + model_type)
    ax.set_xlabel('Axis x')
    ax.set_ylabel('Axis y')
    ax.legend()
    st.pyplot(fig)