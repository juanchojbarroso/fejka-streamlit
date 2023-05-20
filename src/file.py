import streamlit as st
from src.utility import load_dataframe


def upload_file():
    st.sidebar.subheader('Settings')
    st.sidebar.subheader("Upload your data")

    df, columns = None, None

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

    return df, columns
