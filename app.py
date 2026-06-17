import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClubPredict · Membresías Vacacionales",
    page_icon="🏖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #0d2240 50%, #0a1628 100%);
    min-height: 100vh;
}

/* Header hero */
.hero {
    background: linear-gradient(135deg, #1a3a5c 0%, #0d2240 100%);
    border: 1px solid rgba(212, 175, 55, 0.3);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(212,175,55,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #D4AF37;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.65);
    font-weight: 300;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(212,175,55,0.15);
    border: 1px solid rgba(212,175,55,0.4);
    color: #D4AF37;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(212,175,55,0.8);
    margin-bottom: 1rem;
}

/* Result cards */
.result-si {
    background: linear-gradient(135deg, rgba(34,197,94,0.12) 0%, rgba(21,128,61,0.08) 100%);
    border: 1px solid rgba(34,197,94,0.35);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-no {
    background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(153,27,27,0.08) 100%);
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0.5rem 0;
}
.result-si .result-title { color: #4ade80; }
.result-no .result-title { color: #f87171; }
.result-prob {
    font-size: 3.2rem;
    font-weight: 700;
    line-height: 1;
    margin: 0.5rem 0;
}
.result-si .result-prob { color: #22c55e; }
.result-no .result-prob { color: #ef4444; }
.result-label {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.5);
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Metric strip */
.metric-strip {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-box {
    background: rgba(212,175,55,0.07);
    border: 1px solid rgba(212,175,55,0.2);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    flex: 1;
    text-align: center;
}
.metric-val {
    font-size: 1.4rem;
    font-weight: 700;
    color: #D4AF37;
}
.metric-lbl {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.45);
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Streamlit overrides */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stRadio"] label {
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
}
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: white !important;
    border-radius: 8px !important;
}
.stSlider > div > div > div > div {
    background: #D4AF37 !important;
}
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #D4AF37, #b8962e) !important;
    color: #0a1628 !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(212,175,55,0.4) !important;
}
.stRadio > div {
    flex-direction: row !important;
    gap: 1rem !important;
}
hr { border-color: rgba(255,255,255,0.08) !important; }
div[data-testid="stMarkdownContainer"] p {
    color: rgba(255,255,255,0.8);
}

/* Progress bar custom */
.prob-bar-container {
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    height: 10px;
    margin: 1rem 0;
    overflow: hidden;
}
.prob-bar-fill-si {
    background: linear-gradient(90deg, #16a34a, #22c55e);
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
}
.prob-bar-fill-no {
    background: linear-gradient(90deg, #b91c1c, #ef4444);
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
}

/* Section divider */
.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: rgba(212,175,55,0.6);
    margin: 1.5rem 0 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(212,175,55,0.15);
}

/* Tip box */
.tip-box {
    background: rgba(212,175,55,0.06);
    border-left: 3px solid #D4AF37;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin-top: 1rem;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.65);
}
</style>
""", unsafe_allow_html=True)


# ── Load model & encoders ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base, "encoders.pkl"), "rb") as f:
        enc = pickle.load(f)
    with open(os.path.join(base, "model_meta.json"), "r") as f:
        meta = json.load(f)
    return model, enc, meta

model, enc, meta = load_model()


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🏖️ Club Vacacional · IA Predictiva</div>
    <h1 class="hero-title">ClubPredict</h1>
    <p class="hero-subtitle">Predicción inteligente de probabilidad de compra de membresía vacacional</p>
</div>
""", unsafe_allow_html=True)


# ── Model metrics strip ───────────────────────────────────────────────────────
st.markdown(f"""
<div class="metric-strip">
    <div class="metric-box">
        <div class="metric-val">{meta['accuracy']:.0%}</div>
        <div class="metric-lbl">Exactitud</div>
    </div>
    <div class="metric-box">
        <div class="metric-val">{meta['auc']:.2f}</div>
        <div class="metric-lbl">AUC-ROC</div>
    </div>
    <div class="metric-box">
        <div class="metric-val">2 000</div>
        <div class="metric-lbl">Registros</div>
    </div>
    <div class="metric-box">
        <div class="metric-val">10</div>
        <div class="metric-lbl">Variables</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Layout ────────────────────────────────────────────────────────────────────
col_form, col_result = st.columns([1.1, 0.9], gap="large")

with col_form:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Perfil del Prospecto</div>', unsafe_allow_html=True)

    # ── Datos personales ──
    st.markdown('<div class="section-label">Datos Personales</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        nacionalidad = st.selectbox("Nacionalidad", meta["nacionalidades"])
    with c2:
        estado_civil = st.selectbox("Estado Civil", meta["estados_civiles"])

    c3, c4 = st.columns(2)
    with c3:
        edad = st.slider("Edad", min_value=18, max_value=75, value=38)
    with c4:
        tiene_hijos = st.radio("¿Tiene hijos?", ["Sí", "No"], horizontal=True)

    # ── Económico ──
    st.markdown('<div class="section-label">Perfil Económico</div>', unsafe_allow_html=True)
    ingreso_anual = st.slider(
        "Ingreso anual estimado (USD)",
        min_value=15000, max_value=150000,
        value=55000, step=1000,
        format="$%d"
    )

    # ── Visita ──
    st.markdown('<div class="section-label">Datos de la Visita</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        canal_contacto = st.selectbox("Canal de contacto", meta["canales"])
    with c6:
        visitas_previas = st.number_input("Visitas previas al resort", min_value=0, max_value=8, value=1)

    c7, c8 = st.columns(2)
    with c7:
        dias_estancia = st.slider("Días de estancia", min_value=3, max_value=14, value=5)
    with c8:
        calificacion = st.slider("Calificación presentación (1-5)", min_value=1, max_value=5, value=4)

    oferta_especial = st.radio(
        "¿Se le ofreció promoción especial?",
        ["Sí", "No"], horizontal=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("🔍 Analizar Prospecto")


# ── Prediction ────────────────────────────────────────────────────────────────
with col_result:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Resultado del Análisis</div>', unsafe_allow_html=True)

    if predict_btn:
        # Encode
        nac_enc = enc['nac'].transform([nacionalidad])[0]
        civil_enc = enc['civil'].transform([estado_civil])[0]
        canal_enc = enc['canal'].transform([canal_contacto])[0]
        hijos = 1 if tiene_hijos == "Sí" else 0
        oferta = 1 if oferta_especial == "Sí" else 0

        X = np.array([[edad, ingreso_anual, visitas_previas, dias_estancia,
                       hijos, calificacion, oferta,
                       nac_enc, civil_enc, canal_enc]])

        proba = model.predict_proba(X)[0]
        prob_si = proba[1]
        prob_no = proba[0]
        pred = int(model.predict(X)[0])

        if pred == 1:
            st.markdown(f"""
            <div class="result-si">
                <div style="font-size:2.5rem">🏖️</div>
                <div class="result-title">Comprará la Membresía</div>
                <div class="result-prob">{prob_si:.0%}</div>
                <div class="result-label">Probabilidad de compra</div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill-si" style="width:{prob_si*100:.1f}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="tip-box">
            💡 <strong>Recomendación:</strong> Prospecto con alta intención de compra. 
            Avanzar con el proceso de cierre y asignación de ejecutivo de ventas.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-no">
                <div style="font-size:2.5rem">📋</div>
                <div class="result-title">Compra Improbable</div>
                <div class="result-prob">{prob_no:.0%}</div>
                <div class="result-label">Probabilidad de no compra</div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill-no" style="width:{prob_no*100:.1f}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="tip-box">
            💡 <strong>Recomendación:</strong> Considerar seguimiento a largo plazo 
            o ajustar la oferta. Registrar para campaña de nurturing.
            </div>
            """, unsafe_allow_html=True)

        # Desglose de probabilidades
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Desglose de Probabilidades</div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({
            "Resultado": ["✅ Compra membresía", "❌ No compra"],
            "Probabilidad": [f"{prob_si:.1%}", f"{prob_no:.1%}"]
        })
        st.dataframe(prob_df, hide_index=True, use_container_width=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color: rgba(255,255,255,0.3);">
            <div style="font-size:3rem; margin-bottom:1rem;">🔍</div>
            <div style="font-size:0.9rem; letter-spacing:1px; text-transform:uppercase;">
                Completa el perfil del prospecto<br>y presiona <strong style="color:rgba(212,175,55,0.6)">Analizar</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Feature importance
    with st.expander("📊 Importancia de Variables del Modelo"):
        fi = meta["feature_importance"]
        labels = {
            "ingreso_anual": "Ingreso anual",
            "edad": "Edad",
            "nac_enc": "Nacionalidad",
            "dias_estancia": "Días de estancia",
            "calificacion_presentacion": "Calificación presentación",
            "visitas_previas": "Visitas previas",
            "canal_enc": "Canal de contacto",
            "civil_enc": "Estado civil",
            "tiene_hijos": "Tiene hijos",
            "oferta_especial": "Oferta especial",
        }
        fi_df = pd.DataFrame({
            "Variable": [labels.get(k, k) for k in fi],
            "Importancia": list(fi.values())
        }).sort_values("Importancia", ascending=False)
        st.bar_chart(fi_df.set_index("Variable")["Importancia"])


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:rgba(255,255,255,0.25); font-size:0.75rem; padding:1rem 0;">
    ClubPredict · Modelo: Random Forest (class_weight=balanced) · 
    Dataset sintético de 2,000 registros · Desarrollado con Streamlit
</div>
""", unsafe_allow_html=True)
