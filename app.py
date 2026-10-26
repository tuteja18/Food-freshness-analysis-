import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Smart Food Freshness Detection",
    page_icon="🍎",
    layout="wide"
)


# =====================================================
# TITLE
# =====================================================

st.title("🍎 SMART FOOD FRESHNESS DETECTION")
st.subheader("RGB Camera + Gas Sensor Fusion")

st.write(
    "This system combines image-based analysis with gas sensor "
    "readings to estimate food freshness."
)

st.divider()


# =====================================================
# LOAD MODELS
# =====================================================

image_model = joblib.load("image_model.pkl")
gas_model = joblib.load("gas_model.pkl")

image_scaler = joblib.load("image_scaler.pkl")
gas_scaler = joblib.load("gas_scaler.pkl")


# =====================================================
# IMAGE FEATURE EXTRACTION
# =====================================================

def extract_features(image):

    # PIL RGB → OpenCV BGR
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize
    img = cv2.resize(img, (128, 128))

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Hue histogram
    hist_h = cv2.calcHist(
        [hsv], [0], None, [16], [0, 180]
    ).flatten()

    hist_h /= (hist_h.sum() + 1e-6)

    # Saturation histogram
    hist_s = cv2.calcHist(
        [hsv], [1], None, [16], [0, 256]
    ).flatten()

    hist_s /= (hist_s.sum() + 1e-6)

    # Value histogram
    hist_v = cv2.calcHist(
        [hsv], [2], None, [16], [0, 256]
    ).flatten()

    hist_v /= (hist_v.sum() + 1e-6)

    # Mean and standard deviation
    means = hsv.reshape(-1, 3).mean(axis=0)
    stds = hsv.reshape(-1, 3).std(axis=0)

    # Dark pixel ratio
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]

    dark_mask = (
        (v_chan < 90) &
        (s_chan < 120)
    )

    dark_ratio = (
        dark_mask.sum() /
        dark_mask.size
    )

    # Edge density
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    edge_density = (
        edges.sum() /
        (255 * edges.size)
    )

    return np.concatenate([
        hist_h,
        hist_s,
        hist_v,
        means,
        stds,
        [dark_ratio, edge_density]
    ])


# =====================================================
# IMAGE UPLOAD
# =====================================================

st.header("📷 Food Image Analysis")

uploaded_file = st.file_uploader(
    "Upload a fruit image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    # -------------------------------------------------
    # DISPLAY IMAGE
    # -------------------------------------------------

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    # -------------------------------------------------
    # CAMERA PREDICTION
    # -------------------------------------------------

    features = extract_features(image)

    features_scaled = image_scaler.transform(
        [features]
    )

    image_prediction = image_model.predict(
        features_scaled
    )[0]

    image_probabilities = (
        image_model.predict_proba(
            features_scaled
        )[0]
    )

    image_classes = list(
        image_model.classes_
    )

    image_index = image_classes.index(
        image_prediction
    )

    image_confidence = (
        image_probabilities[image_index]
        * 100
    )

    with col2:

        st.subheader("Camera Prediction")

        if image_prediction == "fresh":

            st.success(
                f"🍏 {image_prediction.upper()}"
            )

        else:

            st.error(
                f"⚠️ {image_prediction.upper()}"
            )

        st.metric(
            "Camera Confidence",
            f"{image_confidence:.2f}%"
        )


# =====================================================
# GAS SENSOR INPUT
# =====================================================

st.divider()

st.header("🧪 Gas Sensor Analysis")

st.info(
    "Enter the current gas sensor readings. "
    "These values currently simulate MQ135 + MQ3 measurements."
)


gas_col1, gas_col2, gas_col3 = st.columns(3)


with gas_col1:

    co2 = st.number_input(
        "CO₂ (ppm)",
        min_value=200.0,
        max_value=1300.0,
        value=560.0,
        step=10.0
    )


with gas_col2:

    nh3 = st.number_input(
        "NH₃ (ppm)",
        min_value=2.5,
        max_value=7.5,
        value=4.3,
        step=0.1
    )


with gas_col3:

    mq3 = st.number_input(
        "MQ3 Alcohol (ppm)",
        min_value=20.0,
        max_value=900.0,
        value=160.0,
        step=10.0
    )


# =====================================================
# GAS PREDICTION
# =====================================================

gas_values = np.array([
    [co2, nh3, mq3]
])

gas_values_scaled = gas_scaler.transform(
    gas_values
)

gas_prediction = gas_model.predict(
    gas_values_scaled
)[0]

gas_probabilities = (
    gas_model.predict_proba(
        gas_values_scaled
    )[0]
)

gas_classes = list(
    gas_model.classes_
)

gas_index = gas_classes.index(
    gas_prediction
)

gas_confidence = (
    gas_probabilities[gas_index]
    * 100
)


# DISPLAY GAS RESULT

if gas_prediction == "fresh":

    st.success(
        f"🧪 Gas Prediction: {gas_prediction.upper()}"
    )

else:

    st.error(
        f"🧪 Gas Prediction: {gas_prediction.upper()}"
    )

st.metric(
    "Gas Confidence",
    f"{gas_confidence:.2f}%"
)


# =====================================================
# SENSOR READINGS PANEL
# =====================================================

st.subheader("📡 Current Sensor Readings")

sensor1, sensor2, sensor3 = st.columns(3)

with sensor1:

    st.metric(
        "CO₂",
        f"{co2:.0f} ppm"
    )

with sensor2:

    st.metric(
        "NH₃",
        f"{nh3:.1f} ppm"
    )

with sensor3:

    st.metric(
        "MQ3 Alcohol",
        f"{mq3:.0f} ppm"
    )


# =====================================================
# CAMERA + GAS FUSION
# =====================================================

st.divider()

st.header("🧠 Final Freshness Decision")


if uploaded_file is None:

    st.info(
        "📷 Upload a fruit image to generate "
        "the final fused prediction."
    )

else:

    # Common classes
    classes = list(
        image_model.classes_
    )

    # Align gas probabilities
    gas_probabilities_aligned = np.array([
        gas_probabilities[
            list(gas_model.classes_).index(c)
        ]
        for c in classes
    ])

    # Align image probabilities
    image_probabilities_aligned = np.array([
        image_probabilities[
            list(image_model.classes_).index(c)
        ]
        for c in classes
    ])

    # Equal-weight fusion
    fused_probabilities = (
        image_probabilities_aligned +
        gas_probabilities_aligned
    ) / 2.0

    # Final prediction
    fused_index = np.argmax(
        fused_probabilities
    )

    fused_prediction = classes[
        fused_index
    ]

    fused_confidence = (
        fused_probabilities[
            fused_index
        ] * 100
    )


    # =================================================
    # FINAL RESULT
    # =================================================

    if fused_prediction == "fresh":

        st.success(
            f"🍏 FINAL RESULT: FRESH\n\n"
            f"Fusion Confidence: "
            f"{fused_confidence:.2f}%"
        )

    else:

        st.error(
            f"⚠️ FINAL RESULT: ROTTEN\n\n"
            f"Fusion Confidence: "
            f"{fused_confidence:.2f}%"
        )


    # =================================================
    # COMPARISON
    # =================================================

    st.subheader(
        "📊 Camera vs Gas vs Fused"
    )

    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )


    with result_col1:

        st.metric(
            "📷 Camera",
            image_prediction.upper(),
            f"{image_confidence:.2f}% confidence"
        )


    with result_col2:

        st.metric(
            "🧪 Gas",
            gas_prediction.upper(),
            f"{gas_confidence:.2f}% confidence"
        )


    with result_col3:

        st.metric(
            "🧠 Fused",
            fused_prediction.upper(),
            f"{fused_confidence:.2f}% confidence"
        )


    # =================================================
    # FUSION PROBABILITIES
    # =================================================

    st.subheader(
        "Fusion Probabilities"
    )

    for class_name, probability in zip(
        classes,
        fused_probabilities
    ):

        st.write(
            f"{class_name.capitalize()}: "
            f"{probability * 100:.2f}%"
        )

        st.progress(
            float(probability)
        )


# =====================================================
# MODEL PERFORMANCE
# =====================================================

st.divider()

st.header("📈 Model Performance")

st.write(
    "The following values should be taken from "
    "the accuracy results obtained in your Colab notebook."
)

perf1, perf2, perf3 = st.columns(3)

with perf1:

    st.metric(
        "📷 Camera Model",
        "83.75%"
    )

with perf2:

    st.metric(
        "🧪 Gas Model",
        "97.50%"
    )

with perf3:

    st.metric(
        "🧠 Fusion Model",
        "100.00%"
    )


# =====================================================
# ACCURACY COMPARISON CHART
# =====================================================

st.subheader("Accuracy Comparison")

accuracy_data = {
    "Camera Only": 83.75,
    "Gas Only": 97.50,
    "Camera + Gas Fusion": 100.00
}

st.bar_chart(
    accuracy_data,
    y_label="Accuracy (%)"
)

# =====================================================
# CONFUSION MATRICES
# =====================================================

st.divider()

st.header("📊 Confusion Matrices")

st.write(
    "Confusion matrices show the classification performance "
    "of the camera, gas, and fused models on the test dataset."
)


# =====================================================
# ACTUAL CONFUSION MATRICES FROM COLAB
# =====================================================

camera_cm = np.array([
    [35, 5],
    [8, 32]
])

gas_cm = np.array([
    [39, 1],
    [1, 39]
])

fused_cm = np.array([
    [40, 0],
    [0, 40]
])


# =====================================================
# CREATE DATAFRAMES
# =====================================================

camera_df = __import__("pandas").DataFrame(
    camera_cm,
    index=["Actual Fresh", "Actual Rotten"],
    columns=["Predicted Fresh", "Predicted Rotten"]
)

gas_df = __import__("pandas").DataFrame(
    gas_cm,
    index=["Actual Fresh", "Actual Rotten"],
    columns=["Predicted Fresh", "Predicted Rotten"]
)

fused_df = __import__("pandas").DataFrame(
    fused_cm,
    index=["Actual Fresh", "Actual Rotten"],
    columns=["Predicted Fresh", "Predicted Rotten"]
)


# =====================================================
# DISPLAY THREE MATRICES
# =====================================================

cm_col1, cm_col2, cm_col3 = st.columns(3)


with cm_col1:

    st.subheader("📷 Camera Only")

    st.dataframe(
        camera_df,
        use_container_width=True
    )

    st.caption("Accuracy: 83.75%")


with cm_col2:

    st.subheader("🧪 Gas Only")

    st.dataframe(
        gas_df,
        use_container_width=True
    )

    st.caption("Accuracy: 97.50%")


with cm_col3:

    st.subheader("🧠 Camera + Gas")

    st.dataframe(
        fused_df,
        use_container_width=True
    )

    st.caption("Accuracy: 100.00%")

# =====================================================
# FRESHNESS STATUS & RECOMMENDATION
# =====================================================

st.divider()

st.header("🌱 Freshness Status")

if uploaded_file is not None:

    if fused_prediction == "fresh":

        st.success(
            "🟢 FRESHNESS STATUS: FRESH"
        )

        st.write(
            "The combined camera and gas analysis "
            "indicates that the food appears fresh."
        )

        st.info(
            "💡 Recommendation: The food can be considered "
            "suitable for consumption based on the current "
            "simulation results."
        )

    else:

        st.error(
            "🔴 FRESHNESS STATUS: ROTTEN"
        )

        st.write(
            "The combined camera and gas analysis "
            "indicates possible spoilage."
        )

        st.warning(
            "⚠️ Recommendation: Avoid consuming the food "
            "and inspect or dispose of it appropriately."
        )

    st.metric(
        "Final Fusion Confidence",
        f"{fused_confidence:.2f}%"
    )

else:

    st.info(
        "Upload a food image to display the freshness status."
    )

# =====================================================
# SYSTEM WORKFLOW
# =====================================================

st.divider()

st.header("🔄 System Workflow")

step1, step2, step3, step4 = st.columns(4)

with step1:
    st.subheader("1️⃣ 📷")
    st.write("**Capture Image**")
    st.caption(
        "A fruit image is uploaded and processed by the camera model."
    )

with step2:
    st.subheader("2️⃣ 🧪")
    st.write("**Read Gas Sensors**")
    st.caption(
        "CO₂, NH₃ and MQ3 readings are provided to the gas model."
    )

with step3:
    st.subheader("3️⃣ 🧠")
    st.write("**Fuse Predictions**")
    st.caption(
        "Camera and gas probabilities are combined using equal weighting."
    )

with step4:
    st.subheader("4️⃣ 🌱")
    st.write("**Final Decision**")
    st.caption(
        "The highest fused probability determines the freshness result."
    )


# =====================================================
# PROJECT NOTE
# =====================================================

st.info(
    "ℹ️ Current Version: This dashboard uses real fruit images "
    "and simulated gas-sensor data. The ESP32 + MQ135 + MQ3 "
    "hardware integration will be added in the next stage."
)

# =====================================================
# HOW FUSION WORKS
# =====================================================

st.divider()

st.header("ℹ️ How the System Works")

st.write(
    """
    **Step 1 — Camera:**  
    The uploaded food image is processed using color,
    brightness and edge-based image features.

    **Step 2 — Gas Sensors:**  
    CO₂, NH₃ and MQ3 alcohol readings are processed
    by the gas classification model.

    **Step 3 — Fusion:**  
    The camera and gas model probabilities are combined
    using equal weighting.

    **Step 4 — Final Decision:**  
    The class with the highest fused probability is
    displayed as the final freshness result.
    """
)


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Smart Food Freshness Detection | "
    "RGB Camera + MQ135 + MQ3 Sensor Fusion"
)