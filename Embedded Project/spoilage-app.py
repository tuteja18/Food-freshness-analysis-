import os
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Food Spoilage Time Predictor", page_icon="🍎", layout="wide")

SENSORS = ["mq135", "mq3", "ds18b20_temp", "dht_temp", "humidity"]
IMG_COLS = ["h_mean", "s_mean", "v_mean", "h_std", "s_std", "v_std", "dark_ratio", "edge_density"]
MODEL_PATH = "spoilage_model.pkl"

# Your CSV uses these names -> map them to the names the app uses
COLUMN_RENAMES = {
    "ds18b20_temp_c": "ds18b20_temp",
    "dht_temp_c": "dht_temp",
    "humidity_pct": "humidity",
}


# ---------------------------------------------------------------- features
@st.cache_data(show_spinner=False)
def img_feats(path):
    """8 cheap image features (same idea as your fusion project)."""
    img = cv2.imread(path)
    if img is None:
        return [np.nan] * len(IMG_COLS)
    img = cv2.resize(img, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    flat = hsv.reshape(-1, 3)
    means, stds = flat.mean(axis=0), flat.std(axis=0)
    dark = ((hsv[:, :, 2] < 90) & (hsv[:, :, 1] < 120)).mean()
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)
    edge_density = edges.sum() / (255 * edges.size)
    return [*means, *stds, dark, edge_density]


def parse_timestamps(series):
    """Parse timestamps like 20261015_225003; fall back to pandas' generic parser."""
    s = series.astype(str).str.strip()
    ts = pd.to_datetime(s, format="%Y%m%d_%H%M%S", errors="coerce")
    if ts.isna().any():
        fallback = pd.to_datetime(s[ts.isna()], errors="coerce")
        ts = ts.fillna(fallback)
    return ts


def build_features(df, img_dir, use_images):
    df = df.copy()
    df["timestamp"] = parse_timestamps(df["timestamp"])
    df = df.dropna(subset=["timestamp"])          # drop rows with unreadable timestamps
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["elapsed_h"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() / 3600

    feats = []
    for c in SENSORS:
        df[c] = pd.to_numeric(df[c], errors="coerce").ffill().bfill()
        smooth = df[c].rolling(5, min_periods=1).mean()
        baseline = smooth.iloc[:5].mean()          # first rows = "fresh" baseline
        df[c + "_rel"] = smooth - baseline          # drift from fresh state
        df[c + "_slope"] = smooth.diff(5).fillna(0)  # rate of change
        feats += [c + "_s", c + "_rel", c + "_slope"]
        df[c + "_s"] = smooth

    if use_images and "image_filename" in df.columns:
        rows = [img_feats(os.path.join(img_dir, str(f))) for f in df["image_filename"]]
        img_df = pd.DataFrame(rows, columns=IMG_COLS)
        df = pd.concat([df, img_df], axis=1)
        df[IMG_COLS] = df[IMG_COLS].ffill().bfill()
        feats += IMG_COLS

    # NOTE: elapsed_h is deliberately NOT a feature (it would just memorise this one run).
    return df, feats


def load_csv(csv_file):
    raw = pd.read_csv(csv_file)
    raw = raw.rename(columns=COLUMN_RENAMES)
    missing = [c for c in ["timestamp", *SENSORS] if c not in raw.columns]
    if missing:
        st.error(f"CSV is missing columns: {missing}")
        st.write("Columns found in your file:", raw.columns.tolist())
        st.stop()
    return raw


def status_banner(est):
    if est < 3:
        st.error("🔴 Spoiling imminently")
    elif est < 8:
        st.warning("🟡 Use soon")
    else:
        st.success("🟢 Still good")


# ---------------------------------------------------------------- UI
st.title(" Food Spoilage Time Predictor")
st.caption("MQ135 + MQ3 + temperature/humidity + camera → estimated remaining shelf life")

with st.sidebar:
    mode = st.radio(
        "Mode",
        ["Train a model", "Predict with saved model"],
        help="Train: needs a finished run + the hour the food spoiled. "
             "Predict: uses the saved model on new data, no spoilage hour needed.",
    )

    st.header("1. Data")
    csv_file = st.file_uploader("sensor_log.csv", type=["csv"])
    img_dir = st.text_input("Images folder", "images")

    if mode == "Train a model":
        use_images = st.checkbox("Use image features", value=True)

        st.header("2. Label")
        onset_h = st.number_input(
            "Hours after start when food was actually spoiled",
            min_value=0.1, value=24.0, step=0.5,
            help="Your ground truth: when it looked/smelled rotten. Target = onset − elapsed time.",
        )

        st.header("3. Model")
        split_mode = st.selectbox("Validation", ["Chronological (honest)", "Every 5th row (optimistic)"])
        n_trees = st.slider("Trees", 50, 500, 200, 50)
        train_btn = st.button("Train model", type="primary")
    else:
        st.header("2. Saved model")
        model_path = st.text_input("Model file", MODEL_PATH)

if csv_file is None:
    st.info("Upload your `sensor_log.csv` in the sidebar to begin.")
    st.stop()

raw = load_csv(csv_file)


# ================================================================ PREDICT MODE
if mode == "Predict with saved model":
    if not os.path.exists(model_path):
        st.error(f"Model file `{model_path}` not found. Train a model first (switch mode in the sidebar).")
        st.stop()

    bundle = joblib.load(model_path)
    model, feats_saved = bundle["model"], bundle["feats"]
    uses_images = any(c in IMG_COLS for c in feats_saved)

    if uses_images and "image_filename" not in raw.columns:
        st.error("The saved model uses image features, but this CSV has no `image_filename` column.")
        st.stop()

    df, feats_now = build_features(raw, img_dir, uses_images)
    if df.empty:
        st.error("No rows with a readable timestamp were found.")
        st.stop()

    missing_feats = [f for f in feats_saved if f not in df.columns]
    if missing_feats:
        st.error(f"Could not build these features the model needs: {missing_feats}")
        st.stop()

    st.info(
        "Tip: the model compares readings with the **first 5 rows** of this CSV (the 'fresh' baseline). "
        "Start logging when the food is fresh for the most reliable predictions."
    )

    # ---- sensor trends
    st.header("📡 Sensor trends")
    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(df.set_index("elapsed_h")[["mq135", "mq3"]])
    with c2:
        st.line_chart(df.set_index("elapsed_h")[["ds18b20_temp", "dht_temp", "humidity"]])

    # ---- predictions for every row
    preds = np.clip(model.predict(df[feats_saved].values), 0, None)
    df["predicted_remaining_h"] = preds

    st.header("🔮 Predicted remaining shelf life")
    st.line_chart(df.set_index("elapsed_h")[["predicted_remaining_h"]])

    # median of the last 5 predictions = steadier than a single reading
    est = float(np.median(preds[-5:]))
    st.header("⏳ Latest prediction")
    st.metric("Estimated remaining shelf life (median of last 5 readings)", f"{est:.1f} h")
    status_banner(est)
    st.caption(
        "This comes only from the sensors and camera. No spoilage hour was entered. "
        "Treat it as an estimate: it is only as good as the data the model was trained on."
    )
    st.stop()


# ================================================================ TRAIN MODE
df, feats = build_features(raw, img_dir, use_images)
if df.empty:
    st.error("No rows with a readable timestamp were found.")
    st.stop()

df["remaining_h"] = (onset_h - df["elapsed_h"]).clip(lower=0)
df = df[df["elapsed_h"] <= onset_h].reset_index(drop=True)  # rows after spoilage add nothing

if len(df) < 20:
    st.warning(
        f"Only {len(df)} rows fall within the first {onset_h:.1f} h. "
        "Lower the spoilage hour or collect more data."
    )
    if len(df) < 2:
        st.stop()

# ---- exploration
st.header(" Sensor trends")
c1, c2 = st.columns(2)
with c1:
    st.line_chart(df.set_index("elapsed_h")[["mq135", "mq3"]])
with c2:
    st.line_chart(df.set_index("elapsed_h")[["ds18b20_temp", "dht_temp", "humidity"]])

if use_images and "image_filename" in df.columns:
    st.subheader("📷 Camera timeline")
    idx = np.linspace(0, len(df) - 1, min(5, len(df))).astype(int)
    cols = st.columns(len(idx))
    for col, i in zip(cols, idx):
        p = os.path.join(img_dir, str(df.loc[i, "image_filename"]))
        if os.path.exists(p):
            col.image(p, caption=f"{df.loc[i, 'elapsed_h']:.1f} h", use_container_width=True)

# ---- training
st.header(" Spoilage-time model")

if train_btn:
    X, y = df[feats].values, df["remaining_h"].values
    if split_mode.startswith("Chron"):
        cut = int(len(df) * 0.8)
        tr, te = np.arange(cut), np.arange(cut, len(df))
    else:
        te = np.arange(0, len(df), 5)
        tr = np.setdiff1d(np.arange(len(df)), te)

    model = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=3, random_state=42)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    mae = mean_absolute_error(y[te], pred)

    joblib.dump({"model": model, "feats": feats}, MODEL_PATH)
    st.session_state["bundle"] = {"model": model, "feats": feats}
    st.session_state["eval"] = pd.DataFrame(
        {"elapsed_h": df["elapsed_h"].values[te], "Actual remaining (h)": y[te], "Predicted remaining (h)": pred}
    ).set_index("elapsed_h")
    st.session_state["mae"] = mae
    st.session_state["imp"] = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    st.success(f"Model saved to `{MODEL_PATH}`. Switch to **Predict with saved model** to use it on new data.")

if "bundle" in st.session_state:
    m1, m2 = st.columns([1, 2])
    m1.metric("Test MAE", f"{st.session_state['mae']:.2f} h")
    m2.line_chart(st.session_state["eval"])

    st.subheader("Top features")
    st.bar_chart(st.session_state["imp"].head(10))

    st.header("⏳ Latest prediction (on training data)")
    bundle = st.session_state["bundle"]
    latest = df[bundle["feats"]].iloc[[-1]].values
    est = float(bundle["model"].predict(latest)[0])
    st.metric("Estimated remaining shelf life (latest reading)", f"{est:.1f} h")
    status_banner(est)
else:
    st.info("Set the spoilage hour in the sidebar and click **Train model**.")