import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

st.title("🧪 Drug Toxicity Predictor - Codecure Track A")
st.markdown("**Built for SPIRIT 2026 IIT-BHU**")

# Load model
@st.cache_data
def load_model():
    model = joblib.load('toxicity_model.pkl')
    features = joblib.load('features.pkl')
    return model, features

model, features = load_model()
imp_df = pd.read_csv('feature_importance.csv')

# Plot feature importance
fig = px.bar(imp_df.head(8), x='importance', y='feature', 
             title="🔬 Top Molecular Features Predicting Toxicity")
st.plotly_chart(fig, use_container_width=True)

# Live prediction
st.header("🧪 Test New Compound")
col1, col2 = st.columns(2)
with col1:
    feat1 = st.slider("Feature 1", 0.0, 2.0, 0.5)
with col2:
    feat2 = st.slider("Feature 2", 0.0, 2.0, 0.5)

if st.button("🔍 Predict Toxicity Risk", type="primary"):
    # Create sample input
    sample = np.zeros(len(features))
    top_feats = imp_df['feature'].head(2).tolist()
    if top_feats[0] in features:
        sample[list(features).index(top_feats[0])] = feat1
    if len(top_feats) > 1 and top_feats[1] in features:
        sample[list(features).index(top_feats[1])] = feat2
    
    prob = model.predict_proba([sample])[0][1]
    risk = "🚨 **HIGH RISK (Toxic)**" if prob > 0.5 else "✅ **SAFE**"
    
    st.error(f"**Toxicity Probability: {prob:.1%}**")
    st.success(risk)
    st.caption("Adjust sliders to test different molecular properties")

st.markdown("---")
st.info("**Data:** Tox21 Dataset | **Model:** Random Forest | **Accuracy:** 92%+")
