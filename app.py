import pandas as pd
import streamlit as st

SHEET_ID = "1FDinnIuiVAEOikZAPueHZ_nBFCc87F5mFCL7G9f_Z0c"
DATA_GID = "40828864"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={DATA_GID}"


@st.cache_data
def load_data():
    df = pd.read_csv(CSV_URL)
    df = df.dropna(axis=1, how="all")
    df = df.dropna(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


st.title("Weyland-Yutani Mines Dashboard")

if st.button("Refresh data"):
    load_data.clear()

df = load_data()
mines = [column for column in df.columns if column != "Date"]

st.write(f"Loaded {len(df)} days for {len(mines)} mines: {', '.join(mines)}")
st.dataframe(df)
st.line_chart(df, x="Date", y=mines)