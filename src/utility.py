import pandas as pd
import streamlit as st

@st.cache
def load_dataframe(uploaded_file, clean_data):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        print(e)
        df = pd.read_excel(uploaded_file)

    if clean_data:
        try:
            #Remove missing values
            df = df.dropna()
            #Remove duplicates
            df = df.drop_duplicates()
            #Remove outliers
            df = cap_data(df)
        except Exception as e:
            print(e)      

    columns = list(df.columns)
    columns.append(None)

    return df, columns

def cap_data(df):
    for col in df.columns:
        if (((df[col].dtype)=='float64') | ((df[col].dtype)=='int64')):
            percentiles = df[col].quantile([0.01,0.99]).values
            df[col][df[col] <= percentiles[0]] = percentiles[0]
            df[col][df[col] >= percentiles[1]] = percentiles[1]
        else:
            df[col]=df[col]
    return df