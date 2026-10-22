import io
import json
import math
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

API_URL = "http://127.0.0.1:8000"
TIMEOUT = 180

st.set_page_config(
    page_title="Nano Sentinel | Wafer Intelligence", page_icon="🔬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#f4f7fb;color:#172033}
[data-testid="stSidebar"]{background:#101a2b;border-right:1px solid #1e2b40}
[data-testid="stSidebar"] *{color:#e8eef8!important}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#aebbd0!important}
.brand{font-family:'Space Grotesk',sans-serif;font-size:1.35rem;font-weight:700;letter-spacing:.04em;color:white}
.brand-sub{color:#9aabc4;font-size:.75rem;letter-spacing:.1em}
.eyebrow{color:#4d6b9b;text-transform:uppercase;letter-spacing:.12em;font-size:.72rem;font-weight:700}
.hero{font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:700;letter-spacing:-.04em;color:#142139}
.copy,.muted{color:#718096;font-size:.88rem}
.panel-title{font-family:'Space Grotesk',sans-serif;font-size:1rem;font-weight:700;color:#18263d}
div[data-testid="stMetric"]{background:white;border:1px solid #e4eaf2;border-radius:14px;padding:1rem}
div[data-testid="stMetricValue"]{font-family:'Space Grotesk',sans-serif;color:#172033}
.stButton>button,.stDownloadButton>button{border-radius:10px;font-weight:600;min-height:2.65rem;border:1px solid #d8e1ee}
.stButton>button[kind="primary"]{background:#2459d3;border-color:#2459d3;color:white}
div[data-testid="stFileUploader"]{background:white;border:1px dashed #bdcbe0;border-radius:14px;padding:.6rem}
hr{border-color:#e4eaf2} footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)


def heading(kicker, title, subtitle):
    st.markdown(
        f'<div class="eyebrow">{kicker}</div><div class="hero">{title}</div><div class="copy">{subtitle}</div>', unsafe_allow_html=True)
    st.write("")


def get_api(path, timeout=15):
    return requests.get(f"{API_URL}{path}", timeout=timeout)


def post_file(path, filename, contents):
    return requests.post(f"{API_URL}{path}", files={"file": (filename, contents, "application/octet-stream")}, timeout=TIMEOUT)


def history_data():
    r = get_api("/inspections", 20)
    r.raise_for_status()
    return r.json().get("inspections", [])


def display_percent(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    return f"{number:.2f}%" if math.isfinite(number) else "Unavailable"


def read_wafer(upload):
    contents = upload.getvalue()
    if len(contents) > 10 * 1024 * 1024:
        raise ValueError("File exceeds the 10 MB upload limit.")
    wafer = np.load(io.BytesIO(contents), allow_pickle=False)
    if wafer.ndim != 2 or wafer.size == 0:
        raise ValueError("Expected a non-empty 2D wafer map.")
    if not np.issubdtype(wafer.dtype, np.number) or not np.isfinite(wafer).all():
        raise ValueError("Wafer map must contain finite numeric values.")
    if not np.isin(wafer, [0, 1, 2]).all():
        raise ValueError("Only categorical values 0, 1, and 2 are supported.")
    return contents, wafer


def wafer_plot(wafer, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(wafer, cmap="viridis", vmin=0,
                   vmax=2, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, ticks=[0, 1, 2], fraction=.046, pad=.04)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def explanation_plots(wafer, heatmap):
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.shape != wafer.shape:
        raise ValueError("Heatmap dimensions do not match the wafer map.")
    cols = st.columns(3)
    for i, col in enumerate(cols):
        with col:
            fig, ax = plt.subplots(figsize=(4, 4))
            if i == 0:
                ax.imshow(wafer, cmap="viridis", vmin=0,
                          vmax=2, interpolation="nearest")
                ax.set_title("Original wafer")
            elif i == 1:
                im = ax.imshow(heatmap, cmap="inferno", vmin=0,
                               vmax=1, interpolation="bilinear")
                ax.set_title("Grad-CAM heatmap")
                fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
            else:
                ax.imshow(wafer, cmap="gray", vmin=0,
                          vmax=2, interpolation="nearest")
                im = ax.imshow(heatmap, cmap="jet", alpha=.48,
                               vmin=0, vmax=1, interpolation="bilinear")
                ax.set_title("Activation overlay")
                fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
            ax.axis("off")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


with st.sidebar:
    st.markdown('<div class="brand">◈ NANO SENTINEL</div><div class="brand-sub">INTELLIGENT WAFER INSPECTION</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "WORKSPACE", ["Overview", "New inspection", "Inspection history"])
    st.markdown("---")
    st.markdown("**SYSTEM STATUS**")
    try:
        h = get_api("/health", 5)
        if h.ok:
            data = h.json()
            st.success("Backend online")
            st.caption("MobileNetV2 V2 · 9 classes")
            st.caption(f"Input: {data.get('input_shape', '(96, 96, 3)')}")
        else:
            st.warning("Backend responded with an error")
    except requests.RequestException:
        st.error("Backend offline")
        st.code("python -m uvicorn main:app --reload", language="powershell")
    st.markdown("---")
    st.caption("AI decision support · Human review required")

if page == "Overview":
    heading("MANUFACTURING INTELLIGENCE / WORKSPACE", "Wafer inspection overview",
            "AI-assisted wafer classification, explainability, and inspection traceability.")
    try:
        records = history_data()
        total = len(records)
        review_count = sum(str(x.get("inspection_status", "")).upper() in {
                           "MANUAL_REVIEW", "REVIEW_RECOMMENDED"} for x in records)
        classes = [x.get("predicted_class", "Unknown") for x in records]
        most_common = pd.Series(classes).mode().iloc[0] if classes else "—"
        a, b, c, d = st.columns(4)
        a.metric("Total inspections", total)
        b.metric("Flagged for review", review_count)
        c.metric("Most frequent class", most_common)
        d.metric("Model version", "MobileNetV2 V2")
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown(
                '<div class="panel-title">Recent inspections</div><div class="muted">Latest saved records</div>', unsafe_allow_html=True)
            if records:
                st.dataframe(pd.DataFrame([{
                    "Inspection ID": x.get("inspection_id", "—"),
                    "Class": x.get("predicted_class") or "Unavailable",
                    "Confidence": display_percent(x.get("confidence_percent")),
                    "Severity": x.get("severity_category") or "Unavailable",
                    "Status": str(x.get("inspection_status", "—")).replace("_", " ")
                } for x in records[:5]]), use_container_width=True, hide_index=True)
            else:
                st.info("No inspections saved yet. Start with New inspection.")
        with right:
            st.markdown(
                '<div class="panel-title">Predicted class distribution</div><div class="muted">Across saved inspections</div>', unsafe_allow_html=True)
            if classes:
                fig, ax = plt.subplots(figsize=(5, 3.6))
                pd.Series(classes).value_counts(
                ).sort_values().plot(kind="barh", ax=ax)
                ax.set_xlabel("Inspections")
                ax.set_ylabel("")
                ax.grid(axis="x", alpha=.2)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.caption("The chart will appear after the first inspection.")
    except requests.RequestException:
        st.error("Could not load overview. Check that FastAPI is running.")
    st.markdown("---")
    st.markdown('<div class="panel-title">Inspection workflow</div>',
                unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.markdown("**01 · Upload**  \nLoad a categorical `.npy` wafer map.")
    b.markdown(
        "**02 · Inspect & explain**  \nReview the prediction and request Grad-CAM.")
    c.markdown(
        "**03 · Record & report**  \nBrowse inspection history and PDF reports.")

elif page == "New inspection":
    heading("INSPECTION WORKSPACE / ANALYSIS", "New wafer inspection",
            "Classify a wafer map, review statistics, and visualize model activations.")
    uploaded = st.file_uploader("Choose a NumPy wafer map", type=["npy"],
                                help="0 = background, 1 = normal die, 2 = defective die. Maximum 10 MB.")
    if uploaded:
        try:
            contents, wafer = read_wafer(uploaded)
            die = int(np.count_nonzero(wafer > 0))
            defects = int(np.count_nonzero(wafer == 2))
            ratio = defects / die * 100 if die else 0.0
            left, right = st.columns([1, 1.15])
            with left:
                st.markdown(
                    '<div class="panel-title">Input preview</div>', unsafe_allow_html=True)
                st.caption(f"{uploaded.name} · shape {wafer.shape}")
                wafer_plot(wafer, "Categorical wafer map")
            with right:
                st.markdown(
                    '<div class="panel-title">Map statistics</div>', unsafe_allow_html=True)
                x, y, z = st.columns(3)
                x.metric("Die pixels", f"{die:,}")
                y.metric("Defective pixels", f"{defects:,}")
                z.metric("Defect ratio", f"{ratio:.2f}%")
                st.info(
                    "Severity thresholds are provisional heuristics, not validated manufacturing standards.")
            b1, b2 = st.columns(2)
            inspect = b1.button("Run AI inspection",
                                type="primary", use_container_width=True)
            explain = b2.button("Generate Grad-CAM", use_container_width=True)
            if inspect:
                with st.spinner("Inspecting wafer and generating report..."):
                    try:
                        r = post_file("/inspect", uploaded.name, contents)
                        if r.ok:
                            st.session_state["ns_result"] = r.json()
                            st.session_state["ns_filename"] = uploaded.name
                            st.success("Inspection completed and saved.")
                        else:
                            st.error(
                                f"Inspection failed ({r.status_code}): {r.text}")
                    except requests.RequestException as exc:
                        st.error(f"Backend request failed: {exc}")
            if explain:
                with st.spinner("Calculating Grad-CAM..."):
                    try:
                        r = post_file("/explain", uploaded.name, contents)
                        if r.ok:
                            st.session_state["ns_explanation"] = r.json()
                            st.session_state["ns_filename"] = uploaded.name
                            st.success("Grad-CAM generated.")
                        else:
                            st.error(
                                f"Explanation failed ({r.status_code}): {r.text}")
                    except requests.RequestException as exc:
                        st.error(f"Backend request failed: {exc}")

            if st.session_state.get("ns_filename") == uploaded.name:
                result = st.session_state.get("ns_result")
                if result:
                    st.markdown("---")
                    heading("INSPECTION RESULT", "Prediction summary",
                            "Model output and decision-support indicators.")
                    model = result.get("model", {})
                    severity = result.get("severity", {})
                    decision = result.get("decision", {})
                    a, b, c, d = st.columns(4)
                    a.metric("Predicted class", model.get(
                        "predicted_class") or "Unavailable")
                    b.metric("Confidence", display_percent(
                        model.get("confidence_percent")))
                    c.metric("Severity", severity.get(
                        "severity_category") or "Unavailable")
                    d.metric("Status", str(decision.get(
                        "inspection_status", "—")).replace("_", " "))
                    probs = model.get("probabilities")
                    valid_probs = isinstance(probs, dict) and bool(probs)
                    if valid_probs:
                        try:
                            numeric_probs = {
                                name: float(value) for name, value in probs.items()
                            }
                            valid_probs = (
                                all(math.isfinite(value) and 0 <= value <= 1
                                    for value in numeric_probs.values())
                                and math.isclose(
                                    sum(numeric_probs.values()), 1.0, abs_tol=1e-3
                                )
                            )
                        except (TypeError, ValueError):
                            valid_probs = False
                    if valid_probs:
                        pdf = pd.DataFrame({"Class": list(numeric_probs), "Probability (%)": [
                                           value*100 for value in numeric_probs.values()]})
                        pdf = pdf.sort_values(
                            "Probability (%)", ascending=False)
                        st.markdown(
                            '<div class="panel-title">Class probabilities</div>', unsafe_allow_html=True)
                        st.dataframe(pdf.style.format(
                            {"Probability (%)": "{:.2f}%"}), use_container_width=True, hide_index=True)
                        fig, ax = plt.subplots(figsize=(9, 3.4))
                        pdf.sort_values("Probability (%)").plot.barh(
                            x="Class", y="Probability (%)", ax=ax, legend=False)
                        ax.set_xlim(0, 100)
                        ax.set_xlabel("Probability (%)")
                        ax.grid(axis="x", alpha=.2)
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    else:
                        st.info(
                            "Probability analysis is unavailable because this record does not contain a valid distribution.")
                    st.info(decision.get("recommendation",
                            "Review the model output before corrective action."))
                    st.caption(
                        f'Inspection ID: {result.get("inspection_id", "—")}')
                    st.warning(
                        "Confidence does not guarantee correctness. Severity is provisional.")
                    x, y = st.columns(2)
                    x.download_button("Download JSON record", json.dumps(result, indent=2),
                                      file_name=f'{result.get("inspection_id", "inspection")}.json', mime="application/json", use_container_width=True)
                    try:
                        pr = get_api(
                            f'/inspections/{result["inspection_id"]}/report', 30)
                        if pr.ok:
                            y.download_button("Download PDF report", pr.content,
                                              file_name=f'{result["inspection_id"]}_inspection_report.pdf',
                                              mime="application/pdf", use_container_width=True)
                        else:
                            y.caption("PDF report is not available.")
                    except requests.RequestException:
                        y.caption("Could not retrieve PDF.")
                explanation = st.session_state.get("ns_explanation")
                if explanation:
                    st.markdown("---")
                    heading("EXPLAINABLE AI", "Grad-CAM activation map",
                            "Highlighted regions reflect model activations, not verified physical root causes.")
                    st.markdown(
                        f'**Class:** {explanation.get("predicted_class", "—")} · **Confidence:** {explanation.get("confidence_percent", 0):.2f}% · **Layer:** `{explanation.get("target_layer", "—")}`')
                    explanation_plots(wafer, explanation["heatmap"])
                    st.warning(explanation.get("explanation_note",
                               "Grad-CAM is not proof of defect root cause."))
                    st.download_button("Download Grad-CAM JSON", json.dumps(explanation),
                                       file_name="gradcam_explanation.json", mime="application/json")
        except Exception as exc:
            st.error(f"Could not process wafer map: {exc}")

else:
    heading("RECORDS / TRACEABILITY", "Inspection history",
            "Search saved inspections and retrieve their PDF reports.")
    try:
        records = history_data()
        if not records:
            st.info("No saved inspections yet.")
        else:
            df = pd.DataFrame([{
                "Inspection ID": x.get("inspection_id", "—"),
                "Timestamp (UTC)": x.get("timestamp_utc", "—"),
                "Filename": x.get("filename", "—"),
                "Predicted class": x.get("predicted_class", "—"),
                "Confidence (%)": x.get("confidence_percent", 0),
                "Severity": x.get("severity_category", "—"),
                "Status": str(x.get("inspection_status", "—")).replace("_", " ")
            } for x in records])
            query = st.text_input("Search by ID, filename, class, or status")
            if query.strip():
                mask = df.astype(str).apply(lambda col: col.str.contains(
                    query.strip(), case=False, na=False)).any(axis=1)
                df = df[mask]
            st.dataframe(df, use_container_width=True, hide_index=True)
            ids = [x.get("inspection_id")
                   for x in records if x.get("inspection_id")]
            selected = st.selectbox("Select an inspection for its PDF", ids)
            try:
                r = get_api(f"/inspections/{selected}/report", 30)
                if r.ok:
                    st.download_button("Download selected PDF report", r.content,
                                       file_name=f"{selected}_inspection_report.pdf", mime="application/pdf", type="primary")
                else:
                    st.warning(
                        "No PDF report is available for this inspection.")
            except requests.RequestException as exc:
                st.error(f"Could not retrieve report: {exc}")
    except requests.RequestException as exc:
        st.error(f"History unavailable. Check backend. Details: {exc}")
