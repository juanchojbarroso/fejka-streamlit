import streamlit as st
from src.file import upload_file
from src.ui_model_selector import model_selector
from src.model_creator import create_model

from src.graph_controls import graph_controls


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
            chart_type = st.sidebar.selectbox(label="Select your chart type.",
                                              options=['Scatter plots', 'Density contour',
                                                       'Sunburst', 'Pie Charts', 'Density heatmaps',
                                                       'Histogram', 'Box plots', 'Tree maps',
                                                       'Violin plots', ])  # 'Line plots',

            graph_controls(chart_type=chart_type, df=df,
                           dropdown_options=columns, template = theme_selection)
        else:
            st.subheader("Upload your data to create a chart")

    if link == 'Machine Learning':
        st.header('Machine Learning')
        df, columns=upload_file()

        if columns is not None:
            selected_columns=st.multiselect("Select features", columns)
            data=df[selected_columns]
            target_options=data.columns
            chosen_target=st.selectbox(
                "Select target", (target_options.insert(0, '<select>')), 0)
            if chosen_target != '<select>':
                X=data.loc[:, data.columns != chosen_target]
                X.columns=data.loc[:, data.columns != chosen_target].columns
                y=data[chosen_target]

                default_value_random_state=42
                random_state=st.slider(
                    'Random State', min_value = 1, max_value = 200, value = default_value_random_state)
                test_size=st.selectbox("Select test data size", (0.2, 0.3))

                st.dataframe(df.head(3))

                if len(X) > 1:
                    model_type, model=model_selector()
                    model_btn=st.button('CREATE MODEL')

                    if model_btn:
                        create_model(X, y, model_type, model, test_size,
                                     random_state)
        else:
            st.subheader("Upload your data to create a model")
