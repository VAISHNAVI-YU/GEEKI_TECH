# diagnostic_dashboard.py
# Multimodal Diagnostic Support Dashboard
# ---------------------------------------
# - Upload: Image (X-ray/MRI), Genomic CSV, Clinical Notes (TXT), optional Entities JSON
# - Fuses structured + unstructured data
# - Highlights cross-modal anomalies
# - Suggests differential diagnoses with model confidence
# - Estimates a simple patient-specific prognosis
# NOTE: Educational/demo logic only. Not for clinical use.

import io
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(
    page_title="Multimodal Diagnostic Support Dashboard",
    page_icon="🩺",
    layout="wide"
)

HEADER = """
# 🩺 Multimodal Diagnostic Support Dashboard
This dashboard integrates **medical imaging**, **genomic**, and **clinical text** data to assist clinicians in comprehensive diagnosis and interpretation.  
*Demo logic only — do not use for clinical decisions.*
"""

st.markdown(HEADER)

# -----------------------------
# Utilities & demo defaults
# -----------------------------
def load_image(uploaded):
    try:
        return Image.open(uploaded).convert("RGB")
    except Exception:
        return None

def load_genomic_csv(uploaded) -> pd.DataFrame:
    try:
        df = pd.read_csv(uploaded)
        # Try to keep only reasonable number of columns visible by default
        return df
    except Exception:
        return pd.DataFrame()

def load_text(uploaded) -> str:
    try:
        return uploaded.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

def load_entities_json(uploaded):
    try:
        data = json.load(uploaded)
        if isinstance(data, dict):
            # allow dict or list format
            data = [data]
        return data
    except Exception:
        return []

# Very lightweight NLP: extract a few key entities from clinical text with regex/keywords
def extract_entities_from_text(txt: str):
    txt_low = txt.lower()

    patterns = {
        "tumor_size": [
            r"tumou?r size[:\s]*([\d\.]+\s*(mm|cm))",
            r"mass\s*(measures|measuring)\s*([\d\.]+\s*(mm|cm))",
            r"lesion\s*(size|measure[s]?)[:\s]*([\d\.]+\s*(mm|cm))"
        ],
        "lymph_node_involvement": [
            r"lymph node[s]?\s*(positive|negative|involved|not involved)",
            r"node[s]?\s*(positive|negative)"
        ],
        "histologic_grade": [
            r"grade\s*([123])",
            r"histologic(al)?\s*grade\s*[:\-]?\s*(i{1,3}|\d)"
        ],
        "hormone_receptor_status": [
            r"er[:\s\+\-]*(positive|negative|\+|\-)",
            r"pr[:\s\+\-]*(positive|negative|\+|\-)"
        ],
        "her2_status": [
            r"her2[:\s\+\-]*(positive|negative|not amplified|\+|\-)"
        ],
        "proliferation": [
            r"ki-?67\s*[:>]*\s*(\d+)\s*%"
        ],
        "tumor_growth_rate": [
            r"(fast|slow)[-\s]*(growing|proliferating|progression)",
            r"(rapid|indolent)\s*(growth|proliferation)"
        ]
    }

    out = []
    for entity, pats in patterns.items():
        for pat in pats:
            m = re.search(pat, txt_low, flags=re.I)
            if m:
                value = m.group(1) if len(m.groups()) >= 1 else m.group(0)
                out.append({
                    "entity": entity,
                    "value": value.strip(),
                    "certainty": None,
                    "source": "Clinical note"
                })
                break
    return out

# Very lightweight "genomic features" parsing:
# Try to detect common breast cancer markers/expressions/mutations from a CSV
def summarize_genomics(df: pd.DataFrame):
    if df.empty:
        return {"subtype_hint": None, "mutations": [], "tumor_markers": {}}

    text_blob = " ".join(df.astype(str).fillna("").values.flatten()).lower()

    mutations = []
    for gene in ["pik3ca", "tp53", "gata3", "map3k1", "her2", "erbb2", "brca1", "brca2"]:
        if re.search(rf"\b{gene}\b.*(mut|variant|p\.)", text_blob):
            mutations.append(gene.upper())

    # crude subtype hints
    subtype_hint = None
    if "luminal a" in text_blob or ("er positive" in text_blob and "her2" in text_blob and "negative" in text_blob):
        subtype_hint = "Luminal A-like"
    elif "luminal b" in text_blob or ("er positive" in text_blob and "her2 positive" in text_blob):
        subtype_hint = "Luminal B-like"
    elif "her2" in text_blob and ("positive" in text_blob or "amplified" in text_blob):
        subtype_hint = "HER2-enriched-like"
    elif "triple negative" in text_blob or "basal-like" in text_blob:
        subtype_hint = "Basal-like / TNBC-like"

    tumor_markers = {}
    for marker in ["cea", "ca15-3", "ca 15-3", "ki-67", "ki67"]:
        m = re.search(rf"{marker}\s*[:=]\s*([\d\.]+)", text_blob)
        if m:
            tumor_markers[marker.upper()] = float(m.group(1))

    return {"subtype_hint": subtype_hint, "mutations": mutations, "tumor_markers": tumor_markers}

# Placeholder image "findings": we won't do ML here; we just record presence of large mass from filename hints
def analyze_image_placeholder(image_name: str):
    # if filename contains size (e.g., _31mm), extract it
    m = re.search(r"(\d+)\s*(mm|cm)", (image_name or "").lower())
    size = None
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        size = f"{val} {unit}"
    return {"mass_size_hint": size}

# Fuse entities from clinical text + optional entities JSON + genomic + image hints
def build_unified_profile(clinical_entities, json_entities, genomics_summary, image_summary):
    merged = {}

    def add(ent, val, src):
        if val is None: return
        key = ent
        if key not in merged:
            merged[key] = {"value": val, "sources": [src]}
        else:
            # keep first value, add source
            merged[key]["sources"].append(src)

    for e in clinical_entities:
        add(e["entity"], e.get("value"), "Clinical note")
    for e in json_entities or []:
        add(e.get("entity"), e.get("value"), e.get("source","Entities JSON"))

    add("genomic_subtype_hint", genomics_summary.get("subtype_hint"), "Genomics CSV")
    if genomics_summary.get("mutations"):
        add("mutations", ", ".join(genomics_summary["mutations"]), "Genomics CSV")
    if image_summary.get("mass_size_hint"):
        add("imaging_mass_size_hint", image_summary["mass_size_hint"], "Image filename")

    # Simple normalization for tumor_size fields
    if ("tumor_size" not in merged) and ("imaging_mass_size_hint" in merged):
        add("tumor_size", merged["imaging_mass_size_hint"]["value"], "Imaging→derived")

    # Return as list for display
    profile = []
    for k, v in merged.items():
        profile.append({
            "field": k,
            "value": v["value"],
            "sources": "; ".join(sorted(set(v["sources"])))
        })
    return pd.DataFrame(profile).sort_values("field")

# Rule-based cross-modal anomaly checks
def detect_anomalies(profile_df: pd.DataFrame, genomics_summary):
    anomalies = []

    def get_val(field):
        try:
            return profile_df.loc[profile_df["field"] == field, "value"].values[0]
        except Exception:
            return None

    growth = (get_val("tumor_growth_rate") or "").lower()
    subtype = (get_val("genomic_subtype_hint") or "").lower()
    her2 = (get_val("her2_status") or "").lower()
    hr = (get_val("hormone_receptor_status") or "").lower()
    size = (get_val("tumor_size") or "").lower()

    # Example anomaly 1: fast growth vs Luminal A-like genomic hint
    if "fast" in growth and "luminal a" in subtype:
        anomalies.append({
            "modality_a": "Pathology/Clinical",
            "modality_b": "Genomics",
            "description": "Clinical note suggests fast growth but genomics appears Luminal A-like (typically indolent).",
            "severity": "moderate"
        })

    # Example anomaly 2: HER2- in IHC but genomic hint shows HER2-enriched-like
    if "her2-" in her2 and "her2-enriched" in subtype:
        anomalies.append({
            "modality_a": "IHC",
            "modality_b": "Genomics",
            "description": "IHC reports HER2-, but genomic subtype hint is HER2-enriched-like.",
            "severity": "high"
        })

    # Example anomaly 3: Large tumor size but very low marker values (if present)
    # We only have a few markers if parsed; this is a placeholder rule.
    if size and re.search(r"(\d+(\.\d+)?)\s*(cm|mm)", size):
        val = float(re.search(r"(\d+(\.\d+)?)", size).group(1))
        unit = re.search(r"(cm|mm)", size).group(1)
        size_cm = val/10.0 if unit == "mm" else val
        ca153 = genomics_summary.get("tumor_markers", {}).get("CA 15-3") or genomics_summary.get("tumor_markers", {}).get("CA 15-3".upper())
        if size_cm >= 3.0 and ca153 is not None and ca153 < 20:
            anomalies.append({
                "modality_a": "Imaging",
                "modality_b": "Tumor Markers",
                "description": "Imaging shows ≥3 cm mass but CA 15-3 is low.",
                "severity": "low"
            })

    return anomalies

# Simple differential using heuristic scoring (for demo)
def rank_differential(profile_df: pd.DataFrame):
    # We compute scores for a few breast cancer types based on profile features
    # Normalize to probabilities (softmax)
    dx = {
        "Invasive Ductal Carcinoma": 0.0,
        "Invasive Lobular Carcinoma": 0.0,
        "HER2-enriched Carcinoma": 0.0,
        "Luminal A Carcinoma": 0.0,
        "Basal-like / TNBC": 0.0,
        "Benign Lesion": 0.0,
    }

    def has(field, needle):
        try:
            v = profile_df.loc[profile_df["field"] == field, "value"].values[0]
            return (needle.lower() in str(v).lower())
        except Exception:
            return False

    # Scoring rules (toy example)
    if has("hormone_receptor_status", "er+") or has("hormone_receptor_status", "pr+"):
        dx["Luminal A Carcinoma"] += 1.5
        dx["Invasive Lobular Carcinoma"] += 0.7
        dx["Invasive Ductal Carcinoma"] += 0.5

    if has("her2_status", "her2+"):
        dx["HER2-enriched Carcinoma"] += 1.5
        dx["Invasive Ductal Carcinoma"] += 0.7

    if has("genomic_subtype_hint", "luminal a"):
        dx["Luminal A Carcinoma"] += 1.2

    if has("genomic_subtype_hint", "basal") or has("genomic_subtype_hint", "triple"):
        dx["Basal-like / TNBC"] += 1.5

    if has("mutations", "PIK3CA"):
        dx["Luminal A Carcinoma"] += 0.3
        dx["Invasive Ductal Carcinoma"] += 0.3

    if has("histologic_grade", "3"):
        dx["Basal-like / TNBC"] += 0.5
        dx["Invasive Ductal Carcinoma"] += 0.3

    if has("lymph_node_involvement", "positive"):
        dx["Invasive Ductal Carcinoma"] += 0.4
        dx["HER2-enriched Carcinoma"] += 0.2

    # Benign signal: if everything else is very low info
    if len(profile_df) <= 2:
        dx["Benign Lesion"] += 0.4

    # Convert to probabilities
    scores = np.array(list(dx.values()), dtype=float)
    if np.all(scores == 0):
        scores = np.ones_like(scores) * 1e-3
    probs = np.exp(scores) / np.exp(scores).sum()

    out = []
    for (name, _), p in zip(dx.items(), probs):
        out.append({"Diagnosis": name, "Confidence": float(p)})
    out = sorted(out, key=lambda x: x["Confidence"], reverse=True)
    return pd.DataFrame(out)

# Simple prognosis (toy): map features to 5y DFS / 10y OS
def estimate_prognosis(profile_df: pd.DataFrame):
    base_dfs5 = 0.75
    base_os10 = 0.70

    def get(field):
        try:
            return profile_df.loc[profile_df["field"] == field, "value"].values[0]
        except Exception:
            return None

    hr = (get("hormone_receptor_status") or "").lower()
    her2 = (get("her2_status") or "").lower()
    grade = (get("histologic_grade") or "").lower()
    ln = (get("lymph_node_involvement") or "").lower()
    subtype = (get("genomic_subtype_hint") or "").lower()

    # Adjustments (toy heuristics)
    if "er+" in hr or "pr+" in hr or "luminal a" in subtype:
        base_dfs5 += 0.05
        base_os10 += 0.05
    if "her2+" in her2:
        base_dfs5 -= 0.03
    if "3" in grade:
        base_dfs5 -= 0.05
        base_os10 -= 0.05
    if "positive" in ln:
        base_dfs5 -= 0.06
        base_os10 -= 0.06
    if "basal" in subtype or "triple" in subtype:
        base_dfs5 -= 0.08
        base_os10 -= 0.08

    # Clamp
    dfs5 = float(np.clip(base_dfs5, 0.1, 0.95))
    os10 = float(np.clip(base_os10, 0.1, 0.95))
    return {
        "5-year Disease-Free Survival": dfs5,
        "10-year Overall Survival": os10
    }

def badge(text, color="#111827", bg="#E5E7EB"):
    st.markdown(
        f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        f"background:{bg};color:{color};margin-right:6px;'>{text}</span>",
        unsafe_allow_html=True
    )

# -----------------------------
# Sidebar uploads
# -----------------------------
with st.sidebar:
    st.header("📥 Upload Data")
    up_image = st.file_uploader("Medical Image (X-ray/MRI)", type=["png","jpg","jpeg"])
    up_genomics = st.file_uploader("Genomic CSV", type=["csv"])
    up_notes = st.file_uploader("Clinical Notes (TXT)", type=["txt"])
    up_entities = st.file_uploader("Entities JSON (optional)", type=["json"])
    run_btn = st.button("Run Analysis", use_container_width=True)

# -----------------------------
# Read inputs
# -----------------------------
img = load_image(up_image) if up_image else None
genomic_df = load_genomic_csv(up_genomics) if up_genomics else pd.DataFrame()
clinical_text = load_text(up_notes) if up_notes else ""
entities_json = load_entities_json(up_entities) if up_entities else []

# Show basic sections
col1, col2 = st.columns([1, 1.5])
with col1:
    st.markdown("## 1️⃣ Image Data")
    if img is not None:
        st.image(img, caption="Uploaded Image", use_container_width=True)
        badge("Image loaded", bg="#DCFCE7", color="#065F46")
    else:
        st.info("Upload an image to preview here.")
with col2:
    st.markdown("## 2️⃣ Genomic and Clinical Data")
    st.markdown("### 🧬 Genomic Summary (raw)")
    if not genomic_df.empty:
        st.dataframe(genomic_df, use_container_width=True, height=280)
        badge("Genomics loaded", bg="#E0E7FF", color="#3730A3")
    else:
        st.info("Upload a genomic CSV to see it here.")

    st.markdown("### 🧾 Clinical Notes Preview")
    if clinical_text:
        st.text_area("Clinical Notes", value=clinical_text, height=180)
        badge("Clinical notes loaded", bg="#FDE68A", color="#92400E")
    else:
        st.info("Upload a TXT clinical note to see it here.")

# -----------------------------
# Analysis (runs when button clicked or if all inputs present)
# -----------------------------
should_run = run_btn or (img is not None or not genomic_df.empty or bool(clinical_text))
if should_run:
    with st.spinner("Analyzing patient…"):
        # Extract entities
        clinical_entities = extract_entities_from_text(clinical_text) if clinical_text else []
        if entities_json:
            badge("Entities JSON provided", bg="#DBEAFE", color="#1E3A8A")

        # Summarize genomics
        genomics_summary = summarize_genomics(genomic_df)

        # Image placeholder findings based on filename
        image_summary = analyze_image_placeholder(up_image.name if up_image else "")

        # Build unified profile
        profile_df = build_unified_profile(clinical_entities, entities_json, genomics_summary, image_summary)

        st.markdown("---")
        st.markdown("## 3️⃣ Unified Patient Profile")
        if not profile_df.empty:
            st.dataframe(profile_df, use_container_width=True, height=340)
        else:
            st.warning("No structured profile fields could be derived from your inputs.")

        # Anomalies
        anomalies = detect_anomalies(profile_df, genomics_summary)
        st.markdown("## 4️⃣ ⚠️ Cross-Modal Anomalies")
        if anomalies:
            for a in anomalies:
                sev = a["severity"]
                color = {"low":"#10B981","moderate":"#F59E0B","high":"#EF4444"}.get(sev, "#6B7280")
                st.markdown(
                    f"<div style='border-radius:10px;padding:10px;margin:6px 0;background:rgba(0,0,0,0.03);"
                    f"border-left:8px solid {color};'>"
                    f"<b>{a['modality_a']} ↔ {a['modality_b']}</b><br>{a['description']}<br>"
                    f"<i>Severity: {sev.title()}</i></div>",
                    unsafe_allow_html=True
                )
        else:
            st.success("No cross-modal anomalies detected by the current rules.")

        # Differential diagnoses + confidence
        st.markdown("## 5️⃣ 🧠 Differential Diagnoses (with Confidence)")
        diff_df = rank_differential(profile_df)
        st.dataframe(diff_df, use_container_width=True, height=240)

        # Highlight top prediction
        if not diff_df.empty:
            top = diff_df.iloc[0]
            st.markdown(f"**Predicted Diagnosis:** {top['Diagnosis']}")
            st.progress(float(top["Confidence"]))
            st.markdown(f"**Model Confidence:** {float(top['Confidence'])*100:.2f}%")
            if float(top["Confidence"]) >= 0.9:
                st.success("High confidence — strong model agreement.")
            elif float(top["Confidence"]) >= 0.7:
                st.warning("Moderate confidence — review supporting evidence.")
            else:
                st.error("Low confidence — requires clinician verification.")

        # Prognosis
        st.markdown("## 6️⃣ 💊 Patient-Specific Prognosis")
        prog = estimate_prognosis(profile_df)
        cols = st.columns(len(prog) or 1)
        for i, (k, v) in enumerate(prog.items()):
            cols[i].metric(k, f"{v*100:.1f}%")

        # Simple bar visualization of prognosis
        try:
            fig, ax = plt.subplots(figsize=(5, 2.8))
            ax.barh(list(prog.keys()), list(prog.values()))
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            st.pyplot(fig, use_container_width=False)
        except Exception:
            pass

        # Export current session
        st.markdown("## 7️⃣ 📤 Export (for audit/reproducibility)")
        export_payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "profile": profile_df.to_dict(orient="records"),
            "anomalies": anomalies,
            "differential": diff_df.to_dict(orient="records") if not diff_df.empty else [],
            "prognosis": prog
        }
        st.download_button(
            "Download Session JSON",
            data=json.dumps(export_payload, indent=2),
            file_name="diagnostic_session.json",
            mime="application/json",
            use_container_width=True
        )

st.markdown("---")
st.caption("© Demo build — GPT-5 Thinking. Educational use only. Not a medical device.")
