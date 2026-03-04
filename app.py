import streamlit as st
import pandas as pd
import joblib
import re
import string
from nltk.corpus import stopwords
import nltk
import altair as alt
import os

# ----------------- Configuration -----------------
st.set_page_config(
    page_title="Ticket Classifier",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Custom CSS --------------------
# Injecting custom CSS to match the exact premium theme
st.markdown("""
<style>
/* Sidebar and main background handled by config.toml */
/* Font overrides */
* { font-family: 'Inter', sans-serif !important; }

/* Custom top banner */
.top-banner {
    background-color: #2D2A4A;
    padding: 30px;
    border-radius: 10px;
    margin-bottom: 25px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.banner-title {
    color: #9D72FF;
    font-size: 36px;
    font-weight: bold;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 15px;
}
.banner-subtitle {
    color: #9CA3AF;
    font-size: 14px;
    margin-top: 5px;
}

/* Sidebar Custom Stat Cards */
.sidebar-stat-card {
    background-color: #3B385E;
    border: 1px solid #4B487A;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    margin-bottom: 15px;
}
.sidebar-stat-value {
    color: #9D72FF;
    font-size: 24px;
    font-weight: bold;
    margin: 0;
}
.sidebar-stat-label {
    color: #9CA3AF;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Classification Result Card */
.result-card {
    background-color: #3B385E;
    border: 1px solid #4B487A;
    border-radius: 10px;
    padding: 30px;
    text-align: center;
    margin-top: 10px;
}
.result-label {
    color: #9CA3AF;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 15px;
}
.result-value {
    color: #E2E8F0;
    font-size: 28px;
    font-weight: bold;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 15px;
}

/* Small lists formatting in sidebar */
.sidebar-list {
    list-style-type: none;
    padding: 0;
    font-size: 13px;
    color: #E2E8F0;
}
.sidebar-list li {
    padding: 6px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Override st.text_area container */
div[data-baseweb="textarea"] > div {
    background-color: #2D2A4A !important;
    border-color: #4B487A !important;
    color: #E2E8F0;
}

/* Make primary buttons pop more */
button[kind="primary"] {
    background: linear-gradient(90deg, #7E52FF 0%, #9D72FF 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
}

</style>
""", unsafe_allow_html=True)

# ----------------- Data Loading and Caching -----------------
@st.cache_resource(show_spinner=False)
def download_stopwords():
    try:
        nltk.download('stopwords', quiet=True)
        return set(stopwords.words('english'))
    except:
        return set(['the', 'and', 'is', 'in', 'to', 'of', 'it', 'for', 'a'])

stop_words = download_stopwords()

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return joblib.load("ticket_model.pkl")
    except:
        return None

model = load_model()

@st.cache_data(show_spinner=False)
def load_data():
    try:
        df = pd.read_csv("all_tickets_processed_improved_v3.csv")
        return df
    except:
        return None

df = load_data()

# ----------------- Utility Functions -----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

# Initialize session state for example texts
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

def set_example(text):
    st.session_state.input_text = text

# ----------------- Sidebar -----------------------
st.sidebar.markdown("### 🎫 Ticket Classifier")
st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.sidebar.markdown("#### � Dataset Stats")

total_tickets = len(df) if df is not None else 3806
unique_cats = df['Topic_group'].nunique() if df is not None else 8

st.sidebar.markdown(f"""
<div class="sidebar-stat-card">
    <p class="sidebar-stat-value">{total_tickets:,}</p>
    <p class="sidebar-stat-label">TOTAL TICKETS</p>
</div>
<div class="sidebar-stat-card">
    <p class="sidebar-stat-value">{unique_cats}</p>
    <p class="sidebar-stat-label">CATEGORIES</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("#### 🏷️ All Categories")
st.sidebar.markdown("""
<ul class="sidebar-list">
    <li>🔑 Access</li>
    <li>🛡️ Administrative rights</li>
    <li>👥 HR Support</li>
    <li>💻 Hardware</li>
    <li>📋 Internal Project</li>
    <li>📦 Miscellaneous</li>
    <li>🛒 Purchase</li>
    <li>💾 Storage</li>
</ul>
""", unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("#### 🤖 Model Info")
st.sidebar.markdown("""
<ul class="sidebar-list">
    <li>🔹 Vectorizer: TF-IDF (bigrams)</li>
    <li>🔹 Classifier: LinearSVC</li>
    <li>🔹 Pipeline: sklearn Pipeline</li>
</ul>
""", unsafe_allow_html=True)

# ----------------- Main Body ---------------------
# Top Banner
st.markdown("""
<div class="top-banner">
    <h1 class="banner-title">🎫 Support Ticket Classifier</h1>
    <p class="banner-subtitle">AI-powered classification • Instantly route your support tickets to the right team</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Classify Ticket", "📊 Dataset Overview", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.markdown("### ✏️ Enter Ticket Description")
        
        user_input = st.text_area(
            label="Ticket text", 
            value=st.session_state.input_text,
            height=200, 
            label_visibility="collapsed",
            placeholder="My laptop screen is completely black after the latest update. The monitor does not turn on."
        )
        
        st.markdown("💡 **Try an example:**")
        
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button("Hardware Issue", use_container_width=True):
                set_example("My laptop screen is completely black after the latest update. The monitor does not turn on.")
                st.rerun()
            if st.button("Storage issue", use_container_width=True):
                set_example("I need more space on my network drive to save project files.")
                st.rerun()
        with btn_col2:
            if st.button("Access problem", use_container_width=True):
                set_example("I cannot login to my email account and my password expired.")
                st.rerun()
            if st.button("Purchase request", use_container_width=True):
                set_example("I need to order a new ergonomic mouse and keyboard.")
                st.rerun()
        with btn_col3:
            if st.button("HR Support", use_container_width=True):
                set_example("How do I update my direct deposit information for payroll?")
                st.rerun()
                
        st.markdown("<br>", unsafe_allow_html=True)
        classify_clicked = st.button("🚀 Classify Ticket", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Classification Result")
        
        if classify_clicked:
            if not user_input.strip():
                st.warning("⚠️ Please enter a ticket description before predicting.")
            elif model is None:
                st.error("Model failed to load.")
            else:
                with st.spinner("Processing..."):
                    cleaned = clean_text(user_input)
                    prediction = model.predict([cleaned])
                    result = prediction[0]
                    
                    raw_words = len(user_input.split())
                    clean_words = len(cleaned.split())
                    
                    # Icons map for visual flair
                    icons = {
                        "Hardware": "💻", "Access": "🔑", "HR Support": "👥",
                        "Administrative rights": "🛡️", "Internal Project": "📋",
                        "Miscellaneous": "📦", "Purchase": "🛒", "Storage": "💾"
                    }
                    icon = icons.get(result, "🎯")
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <p class="result-label">PREDICTED CATEGORY</p>
                        <div class="result-value">{icon} {result}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("📄 **Ticket Preview:**")
                    st.markdown(f"<p style='font-style: italic; color: #9CA3AF; border-left: 2px solid #9D72FF; padding-left: 10px;'>{user_input}</p>", unsafe_allow_html=True)
                    
                    stat_col1, stat_col2 = st.columns(2)
                    with stat_col1:
                        st.markdown("<p style='font-size: 14px; margin-bottom: 0px;'>Raw Words</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size: 24px; font-weight: bold;'>{raw_words}</p>", unsafe_allow_html=True)
                    with stat_col2:
                        st.markdown("<p style='font-size: 14px; margin-bottom: 0px;'>After Cleaning</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size: 24px; font-weight: bold;'>{clean_words}</p>", unsafe_allow_html=True)
        else:
            # Empty state
            st.markdown("""
            <div class="result-card" style="opacity: 0.5;">
                <p class="result-label">PREDICTED CATEGORY</p>
                <div class="result-value">等待输入...</div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 📊 Dataset Overview")
    if df is not None:
        category_counts = df['Topic_group'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        
        chart = alt.Chart(category_counts).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X('Count:Q', title='Number of Tickets'),
            y=alt.Y('Category:N', sort='-x', title='Ticket Category'),
            color=alt.Color('Category:N', legend=None, scale=alt.Scale(scheme='purples')),
            tooltip=['Category', 'Count']
        ).properties(
            height=400
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.warning("Dataset not available.")

with tab3:
    st.markdown("### ℹ️ About this Project")
    st.write("This tool uses Natural Language Processing and Machine Learning to automatically route unstructured text logs into designated categories to alleviate manual support burden.")
