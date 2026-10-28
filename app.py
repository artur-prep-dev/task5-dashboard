import io
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF
from plotly.colors import qualitative
from scipy import stats as scipy_stats

DATA_LINK = "https://docs.google.com/spreadsheets/d/1FDinnIuiVAEOikZAPueHZ_nBFCc87F5mFCL7G9f_Z0c/export?format=csv&gid=40828864"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_LINK)
    df = df.dropna(axis=1, how="all")
    df = df.dropna(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.reset_index(drop=True)
    return df

def calculate_statistics(df, columns):
    stats = pd.DataFrame()
    stats["Mean"] = df[columns].mean()
    stats["Std"] = df[columns].std()
    stats["Median"] = df[columns].median()
    stats["IQR"] = df[columns].quantile(0.75) - df[columns].quantile(0.25)
    return stats.round(1)

def detect_iqr(series, k):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - k * iqr) | (series > q3 + k * iqr)

def detect_zscore(series, threshold):
    z = (series - series.mean()) / series.std()
    return z.abs() > threshold

def detect_moving_average(series, window, percent):
    moving_average = series.rolling(window, center=True, min_periods=1).mean()
    distance = (series - moving_average).abs() / moving_average * 100
    return distance > percent

def detect_grubbs(series, alpha):
    values = series.copy()
    outliers = pd.Series(False, index=series.index)
    while len(values) > 2:
        n = len(values)
        mean = values.mean()
        std = values.std()
        if std == 0:
            break
        distances = (values - mean).abs()
        farthest = distances.idxmax()
        g = distances[farthest] / std
        t = scipy_stats.t.ppf(1 - alpha / (2 * n), n - 2)
        g_critical = (n - 1) / np.sqrt(n) * np.sqrt(t ** 2 / (n - 2 + t ** 2))
        if g > g_critical:
            outliers[farthest] = True
            values = values.drop(farthest)
        else:
            break
    return outliers

def find_anomalies(series, settings):
    result = pd.DataFrame()
    result["IQR"] = detect_iqr(series, settings["iqr_k"])
    result["Z-score"] = detect_zscore(series, settings["z_threshold"])
    result["Moving average"] = detect_moving_average(series, settings["ma_window"], settings["ma_percent"])
    result["Grubbs"] = detect_grubbs(series, settings["grubbs_alpha"])
    return result

def methods_list(row):
    found = []
    for method in row.index:
        if row[method]:
            found.append(method)
    return ", ".join(found)

def anomaly_groups(flags):
    days = list(flags.index[flags.any(axis=1)])
    groups = []
    for day in days:
        if len(groups) > 0 and day - groups[-1][-1] == 1:
            groups[-1].append(day)
        else:
            groups.append([day])
    return groups

def trend_line(series, degree):
    x = np.arange(len(series))
    coefficients = np.polyfit(x, series.values, degree)
    return np.polyval(coefficients, x)

def make_chart(df, chart_type, shown, mines, degree, anomalies, colors):
    fig = go.Figure()
    if chart_type == "Stacked":
        for mine in mines:
            fig.add_trace(go.Bar(x=df["Date"], y=df[mine], name=mine, marker_color=colors[mine]))
        fig.update_layout(barmode="stack")
        highlighted = ["Total"]
    else:
        for column in shown:
            if chart_type == "Line":
                fig.add_trace(go.Scatter(x=df["Date"], y=df[column], mode="lines", name=column, line=dict(color=colors[column])))
            else:
                fig.add_trace(go.Bar(x=df["Date"], y=df[column], name=column, marker_color=colors[column]))
        fig.update_layout(barmode="group")
        highlighted = shown
    for column in highlighted:
        trend = trend_line(df[column], degree)
        fig.add_trace(go.Scatter(x=df["Date"], y=trend, mode="lines", name=column + ": trend", line=dict(color=colors[column], dash="dash", width=3)))
        is_anomaly = anomalies[column].any(axis=1)
        fig.add_trace(go.Scatter(x=df.loc[is_anomaly, "Date"], y=df.loc[is_anomaly, column], mode="markers", name=column + ": anomalies", marker=dict(color="red", size=12, symbol="circle-open", line=dict(width=3))))
    fig.update_layout(height=550, xaxis_title="Date", yaxis_title="Output", legend=dict(orientation="h", y=-0.2))
    return fig

def chart_image(df, chart_type, shown, mines, degree, anomalies, colors):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if chart_type == "Stacked":
        bottom = np.zeros(len(df))
        for mine in mines:
            ax.bar(df["Date"], df[mine], bottom=bottom, width=1, color=colors[mine], label=mine)
            bottom = bottom + df[mine].values
        highlighted = ["Total"]
    elif chart_type == "Bar":
        width = 0.8 / len(shown)
        for i in range(len(shown)):
            column = shown[i]
            shift = pd.Timedelta(days=width * i - 0.4)
            ax.bar(df["Date"] + shift, df[column], width=width, align="edge", color=colors[column], label=column)
        highlighted = shown
    else:
        for column in shown:
            ax.plot(df["Date"], df[column], color=colors[column], label=column)
        highlighted = shown
    for column in highlighted:
        ax.plot(df["Date"], trend_line(df[column], degree), color=colors[column], linestyle="--", linewidth=2)
        is_anomaly = anomalies[column].any(axis=1)
        ax.scatter(df.loc[is_anomaly, "Date"], df.loc[is_anomaly, column], s=80, facecolors="none", edgecolors="red", linewidths=2, zorder=5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Output")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer

def clean(text):
    return str(text).encode("latin-1", "replace").decode("latin-1")

def pdf_heading(pdf, text):
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, clean(text), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)

def pdf_text(pdf, text):
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, clean(text), new_x="LMARGIN", new_y="NEXT")

def pdf_table(pdf, header, rows):
    pdf.set_font("Helvetica", "", 9)
    with pdf.table() as table:
        row = table.row()
        for item in header:
            row.cell(clean(item))
        for values in rows:
            row = table.row()
            for item in values:
                row.cell(clean(item))
    pdf.set_font("Helvetica", "", 10)

def pdf_chart(pdf, title, image):
    if pdf.get_y() > 170:
        pdf.add_page()
    pdf_heading(pdf, title)
    pdf.image(image, w=pdf.epw)

def make_pdf(df, columns, mines, stats, anomalies, settings, chart_type, shown, degree, colors):
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "Weyland-Yutani Mines Report", new_x="LMARGIN", new_y="NEXT")
    start = df["Date"].min().date()
    end = df["Date"].max().date()
    pdf_text(pdf, f"Report date: {date.today()}\nPeriod: {start} - {end} ({len(df)} days)\nMines: {', '.join(mines)}")

    pdf_heading(pdf, "Anomaly detection settings")
    rows = []
    rows.append(["IQR rule", "k", settings["iqr_k"]])
    rows.append(["Z-score", "threshold", settings["z_threshold"]])
    rows.append(["Moving average", "window, days", settings["ma_window"]])
    rows.append(["Moving average", "max distance, %", settings["ma_percent"]])
    rows.append(["Grubbs' test", "alpha", settings["grubbs_alpha"]])
    pdf_table(pdf, ["Method", "Parameter", "Value"], rows)

    pdf_heading(pdf, "Statistics")
    rows = []
    for column in columns:
        rows.append([column, stats.loc[column, "Mean"], stats.loc[column, "Std"], stats.loc[column, "Median"], stats.loc[column, "IQR"]])
    pdf_table(pdf, ["", "Mean", "Std", "Median", "IQR"], rows)

    pdf_heading(pdf, "Anomalies found (number of days)")
    rows = []
    for column in columns:
        counts = anomalies[column].sum()
        rows.append([column, counts["IQR"], counts["Z-score"], counts["Moving average"], counts["Grubbs"]])
    pdf_table(pdf, ["", "IQR", "Z-score", "Moving average", "Grubbs"], rows)

    pdf.add_page()
    pdf_chart(pdf, f"Chart: {chart_type}, trendline degree {degree}", chart_image(df, chart_type, shown, mines, degree, anomalies, colors))
    for column in columns:
        pdf_chart(pdf, f"Chart: {column}", chart_image(df, "Line", [column], mines, degree, anomalies, colors))

    pdf.add_page()
    pdf_heading(pdf, "Anomalies in detail")
    number = 0
    for column in columns:
        flags = anomalies[column]
        median = df[column].median()
        for group in anomaly_groups(flags):
            number = number + 1
            values = df.loc[group, column]
            if values.mean() > median:
                kind = "Spike"
                peak = values.max()
            else:
                kind = "Drop"
                peak = values.min()
            change = (peak - median) / median * 100
            first = df.loc[group[0], "Date"].date()
            last = df.loc[group[-1], "Date"].date()
            if len(group) == 1:
                dates = f"{first} (1 day)"
            else:
                dates = f"{first} - {last} ({len(group)} days)"
            methods = methods_list(flags.loc[group].any())
            if pdf.get_y() > 230:
                pdf.add_page()
            pdf_heading(pdf, f"Anomaly {number}: {kind} at {column}")
            pdf_text(pdf, f"Dates: {dates}\nPeak value: {peak:.0f} (typical level: {median:.0f}, change: {change:+.1f}%)\nFound by: {methods}")
            rows = []
            for day in group:
                rows.append([df.loc[day, "Date"].date(), df.loc[day, column], methods_list(flags.loc[day])])
            pdf_table(pdf, ["Date", "Value", "Found by"], rows)
    if number == 0:
        pdf_text(pdf, "No anomalies found with the current settings.")
    return bytes(pdf.output())

st.set_page_config(page_title="Weyland-Yutani Mines", page_icon="⛏️", layout="wide")

st.sidebar.title("Weyland-Yutani Mines")
if st.sidebar.button("Refresh data"):
    load_data.clear()

st.sidebar.header("Anomaly detection settings")
settings = {}
settings["iqr_k"] = st.sidebar.slider("IQR rule: k", 0.5, 3.0, 1.5, 0.1)
settings["z_threshold"] = st.sidebar.slider("Z-score: threshold", 1.0, 5.0, 2.5, 0.1)
settings["ma_window"] = st.sidebar.slider("Moving average: window, days", 3, 45, 21)
settings["ma_percent"] = st.sidebar.slider("Moving average: max distance, %", 5, 100, 40)
settings["grubbs_alpha"] = st.sidebar.selectbox("Grubbs' test: alpha", [0.01, 0.05, 0.1], index=1)

df = load_data()
mines = list(df.columns)
mines.remove("Date")
df["Total"] = df[mines].sum(axis=1)
columns = mines + ["Total"]

colors = {}
for i in range(len(columns)):
    colors[columns[i]] = qualitative.Plotly[i % 10]

anomalies = {}
for column in columns:
    anomalies[column] = find_anomalies(df[column], settings)

stats = calculate_statistics(df, columns)
start = df["Date"].min().date()
end = df["Date"].max().date()

st.title("Weyland-Yutani Mines Dashboard")
st.caption(f"Daily resource output from {start} to {end}. Mines: {', '.join(mines)}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Mines", len(mines))
col2.metric("Days", len(df))
col3.metric("Average total output per day", f"{stats.loc['Total', 'Mean']:.0f}")
col4.metric("Anomaly days in Total", int(anomalies["Total"].any(axis=1).sum()))

tab1, tab2, tab3, tab4 = st.tabs(["Chart", "Statistics", "Anomalies", "Data"])

with tab1:
    chart_type = st.radio("Chart type", ["Line", "Bar", "Stacked"], horizontal=True)
    degree = st.selectbox("Trendline: polynomial degree", [1, 2, 3, 4])
    if chart_type == "Stacked":
        st.caption("Stacked chart shows all mines on top of each other. Trendline and anomalies are shown for Total.")
        shown = mines
    else:
        shown = st.multiselect("Show", columns, default=mines)
    if len(shown) > 0:
        st.plotly_chart(make_chart(df, chart_type, shown, mines, degree, anomalies, colors))
    else:
        st.info("Choose at least one mine to show.")

with tab2:
    st.subheader("Statistics for each mine and total")
    st.dataframe(stats)

with tab3:
    st.subheader("Anomalies found (number of days)")
    summary = pd.DataFrame()
    for column in columns:
        summary[column] = anomalies[column].sum()
    st.dataframe(summary.T)
    selected = st.selectbox("Show anomaly days for", columns)
    flags = anomalies[selected]
    is_anomaly = flags.any(axis=1)
    details = df.loc[is_anomaly, ["Date", selected]].copy()
    details["Found by"] = flags[is_anomaly].apply(methods_list, axis=1)
    details["Date"] = details["Date"].dt.date
    st.dataframe(details, hide_index=True)

with tab4:
    st.dataframe(df, hide_index=True, column_config={"Date": st.column_config.DateColumn(format="YYYY-MM-DD")})

st.sidebar.header("PDF report")
report_shown = shown
if len(report_shown) == 0:
    report_shown = mines
if st.sidebar.button("Generate PDF report"):
    pdf_bytes = make_pdf(df, columns, mines, stats, anomalies, settings, chart_type, report_shown, degree, colors)
    st.sidebar.download_button("Download PDF", pdf_bytes, file_name="mines_report.pdf", mime="application/pdf")