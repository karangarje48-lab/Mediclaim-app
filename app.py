import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Mediclaim Cost Predictor", page_icon="🏥", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "mediclaim_raw.csv")
MODEL_PATH = os.path.join(BASE, "model", "insurance_model_pipeline.pkl")
REGIONS = ["northeast", "northwest", "southeast", "southwest"]


# ---------------------------------------------------------------- data + model
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning steps as Mediclaim.ipynb."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    for c in ["age", "bmi", "children", "charges"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["sex", "smoker", "region"]:
        df[c] = df[c].astype("string").str.strip().str.lower()
    df["sex"] = df["sex"].replace({"m": "male", "f": "female"})
    df["smoker"] = df["smoker"].replace({"y": "yes", "n": "no"})
    df["region"] = df["region"].str.replace(" ", "", regex=False)
    df.loc[~df["sex"].isin(["male", "female"]), "sex"] = np.nan
    df.loc[~df["smoker"].isin(["yes", "no"]), "smoker"] = np.nan
    df.loc[~df["region"].isin(REGIONS), "region"] = np.nan
    df.loc[(df["age"] < 0) | (df["age"] > 100), "age"] = np.nan
    df.loc[(df["bmi"] <= 0) | (df["bmi"] > 80), "bmi"] = np.nan
    df.loc[df["children"] < 0, "children"] = np.nan
    df.loc[df["charges"] <= 0, "charges"] = np.nan
    df = df.dropna(subset=["charges"]).reset_index(drop=True)
    for c in ["sex", "smoker", "region"]:
        df[c] = df[c].astype(object)
    return df


def to_model_input(X: pd.DataFrame) -> pd.DataFrame:
    return X.astype(object).fillna(np.nan)


@st.cache_data
def load_data():
    raw = pd.read_csv(DATA_PATH)
    return raw, clean_data(raw)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_metrics(_model, n_rows: int):
    _, df = load_data()
    X, y = df.drop(columns=["charges"]), df["charges"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pred = _model.predict(to_model_input(X_test))
    mse = mean_squared_error(y_test, pred)
    return {
        "R2": r2_score(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mse**0.5,
        "y_test": y_test.values,
        "y_pred": pred,
    }


def predict_one(model, row: dict) -> float:
    return float(model.predict(to_model_input(pd.DataFrame([row])))[0])


raw_df, df = load_data()
model = load_model()
metrics = get_metrics(model, len(df))

# ---------------------------------------------------------------- sidebar
st.sidebar.title("🏥 Mediclaim App")
page = st.sidebar.radio(
    "Navigate", ["Predict Charges", "Data Explorer", "EDA Dashboard", "Model Performance"]
)
st.sidebar.caption(f"{len(df):,} clean records • Random Forest")

# ---------------------------------------------------------------- predict
if page == "Predict Charges":
    st.title("Insurance Charges Predictor")
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 100, 35)
        bmi = st.number_input("BMI", 10.0, 70.0, 27.0, step=0.1)
        children = st.slider("Children", 0, 10, 0)
    with c2:
        sex = st.selectbox("Sex", ["female", "male"])
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", REGIONS)

    if st.button("Predict", type="primary"):
        row = dict(age=float(age), sex=sex, bmi=float(bmi), children=float(children),
                   smoker=smoker, region=region)
        pred = predict_one(model, row)
        st.success(f"Estimated annual charges: **${pred:,.2f}**")

        m1, m2, m3 = st.columns(3)
        m1.metric("Dataset average", f"${df['charges'].mean():,.0f}")
        m2.metric("Dataset median", f"${df['charges'].median():,.0f}")
        m3.metric("Percentile", f"{(df['charges'] < pred).mean() * 100:.0f}th")
        st.caption(f"Typical model error (MAE): ±${metrics['MAE']:,.0f}")

        alt = dict(row, smoker="no" if smoker == "yes" else "yes")
        alt_pred = predict_one(model, alt)
        st.info(f"What-if as a {'non-smoker' if smoker == 'yes' else 'smoker'}: "
                f"**${alt_pred:,.2f}** ({alt_pred - pred:+,.0f})")

        ages = list(range(18, 71, 2))
        curve = pd.DataFrame({"age": ages,
                              "charges": [predict_one(model, dict(row, age=float(a))) for a in ages]})
        fig = px.line(curve, x="age", y="charges", title="Predicted charges by age (same profile)")
        fig.add_scatter(x=[age], y=[pred], mode="markers", marker=dict(size=12, color="red"), name="You")
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Batch prediction")
    up = st.file_uploader("Upload CSV with columns: age, sex, bmi, children, smoker, region", type="csv")
    if up:
        batch = clean_data(pd.read_csv(up).assign(charges=1.0)).drop(columns=["charges"])
        batch["predicted_charges"] = model.predict(to_model_input(batch)).round(2)
        st.dataframe(batch, width="stretch")
        st.download_button("Download predictions", batch.to_csv(index=False).encode(),
                           "predictions.csv", "text/csv")

# ---------------------------------------------------------------- data
elif page == "Data Explorer":
    st.title("Data Explorer")
    t1, t2 = st.tabs(["Cleaned data", "Raw data quality"])
    with t1:
        f1, f2, f3 = st.columns(3)
        smk = f1.multiselect("Smoker", ["yes", "no"], default=["yes", "no"])
        reg = f2.multiselect("Region", REGIONS, default=REGIONS)
        a_min, a_max = f3.slider("Age range", 18, 100, (18, 100))
        view = df[df["smoker"].isin(smk) | df["smoker"].isna()]
        view = view[view["region"].isin(reg) | view["region"].isna()]
        view = view[view["age"].between(a_min, a_max) | view["age"].isna()]
        st.write(f"{len(view):,} rows")
        st.dataframe(view, width="stretch")
        st.download_button("Download filtered CSV", view.to_csv(index=False).encode(),
                           "mediclaim_filtered.csv", "text/csv")
    with t2:
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw rows", f"{len(raw_df):,}")
        c2.metric("Duplicate rows", int(raw_df.duplicated().sum()))
        c3.metric("Rows after cleaning", f"{len(df):,}")
        st.write("Missing values before vs after cleaning")
        st.dataframe(pd.DataFrame({"raw_missing": raw_df.isna().sum(), "clean_missing": df.isna().sum()}))

# ---------------------------------------------------------------- EDA
elif page == "EDA Dashboard":
    st.title("Exploratory Data Analysis")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg charges", f"${df['charges'].mean():,.0f}")
    k2.metric("Avg age", f"{df['age'].mean():.1f}")
    k3.metric("Avg BMI", f"{df['bmi'].mean():.1f}")
    k4.metric("Smokers", f"{(df['smoker'] == 'yes').mean() * 100:.0f}%")

    a, b = st.columns(2)
    a.plotly_chart(px.histogram(df, x="charges", nbins=50, marginal="box",
                                title="Charges distribution"), width="stretch")
    b.plotly_chart(px.box(df.dropna(subset=["smoker"]), x="smoker", y="charges", color="smoker",
                          title="Charges by smoking status"), width="stretch")
    c, d = st.columns(2)
    c.plotly_chart(px.scatter(df, x="age", y="charges", color="smoker", opacity=0.6,
                              title="Age vs charges"), width="stretch")
    d.plotly_chart(px.scatter(df, x="bmi", y="charges", color="smoker", opacity=0.6,
                              title="BMI vs charges"), width="stretch")
    e, f = st.columns(2)
    e.plotly_chart(px.bar(df.groupby("region", as_index=False)["charges"].mean(),
                          x="region", y="charges", title="Average charges by region"),
                   width="stretch")
    f.plotly_chart(px.imshow(df.select_dtypes("number").corr(), text_auto=".2f",
                             color_continuous_scale="RdBu_r", title="Correlation heatmap"),
                   width="stretch")

# ---------------------------------------------------------------- model
else:
    st.title("Model Performance")
    st.caption("Random Forest • 80/20 train-test split • random_state=42")
    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{metrics['R2']:.3f}")
    m2.metric("MAE", f"${metrics['MAE']:,.0f}")
    m3.metric("RMSE", f"${metrics['RMSE']:,.0f}")

    res = pd.DataFrame({"actual": metrics["y_test"], "predicted": metrics["y_pred"]})
    res["residual"] = res["actual"] - res["predicted"]
    a, b = st.columns(2)
    fig = px.scatter(res, x="actual", y="predicted", opacity=0.6, title="Actual vs predicted")
    lim = [res.actual.min(), res.actual.max()]
    fig.add_scatter(x=lim, y=lim, mode="lines", name="Perfect fit", line=dict(dash="dash"))
    a.plotly_chart(fig, width="stretch")
    b.plotly_chart(px.histogram(res, x="residual", nbins=40, title="Residuals"), width="stretch")

    imp = pd.DataFrame({
        "feature": model.named_steps["preprocessor"].get_feature_names_out(),
        "importance": model.named_steps["model"].feature_importances_,
    }).sort_values("importance")
    imp["feature"] = imp["feature"].str.replace(r"^(num|cat)__", "", regex=True)
    st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
                           title="Feature importance"), width="stretch")
