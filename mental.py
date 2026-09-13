import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. APP CONFIGURATION & SETTINGS
# ==========================================
st.set_page_config(
    page_title="Tech Mental Health Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set global plotting style for charts rendered inside the app
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)

# ==========================================
# 2. DATA CLEANING PIPELINE (Cached for speed)
# ==========================================
@st.cache_data
def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return None
        
    clean_df = df.copy()
    
    # 1. Sanitize outlier values out of the Age feature
    clean_df.loc[(clean_df['Age'] < 18) | (clean_df['Age'] > 75), 'Age'] = np.nan
    clean_df['Age'] = clean_df['Age'].fillna(clean_df['Age'].median()).astype(int)
    
    # 2. Map messy string inputs into three explicit categorical values
    def clean_gender_strings(text):
        if pd.isna(text):
            return 'Unknown'
        text = str(text).lower().strip()
        males = ['male', 'm', 'male-ish', 'maile', 'mal', 'male (cis)', 'make', 'guy (-ish) ^_^', 'man', 'malr']
        females = ['female', 'f', 'woman', 'cis female', 'femail', 'femake']
        
        if text in males:
            return 'Male'
        elif text in females:
            return 'Female'
        else:
            return 'Non-Binary / Diverse'

    clean_df['Gender'] = clean_df['Gender'].apply(clean_gender_strings)
    
    # 3. Handle structural null spaces fallback properties safely
    clean_df['self_employed'] = clean_df['self_employed'].fillna('Unknown')
    clean_df['work_interfere'] = clean_df['work_interfere'].fillna('Don\'t know')
    clean_df['state'] = clean_df['state'].fillna('Not US Resident')
    
    return clean_df

# ==========================================
# AUTOMATED ABSOLUTE PATH RESOLUTION (FIXED)
# ==========================================
# Finds the exact folder directory where app.py is currently saved
current_directory = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(current_directory, 'survey.csv')

# Load dataset using our absolute file path
df_clean = load_and_clean_data(csv_file_path)

# Error handler backup fallback path check
if df_clean is None:
    st.error("🚨 'mental_health_survey.csv' not found! Please ensure your dataset is placed in the exact same folder as this app.py file.")
    st.stop()

# ==========================================
# 3. SIDEBAR NAVIGATION & INTERACTIVE FILTERS
# ==========================================
st.sidebar.title("🧠 Dashboard Navigation")
app_mode = st.sidebar.radio("Go to Section:", ["Project Overview", "Interactive Workforce Insights"])

# Global Sidebar Data Filtering Panel
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Workspace Filters")

# Filter 1: Geographic Filter Selection
all_countries = ["All Countries"] + list(df_clean['Country'].unique())
selected_country = st.sidebar.selectbox("Filter by Country Location:", all_countries)

# Filter 2: Company Sizing Filter Selection
all_sizes = ["All Sizes", "1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]
selected_size = st.sidebar.selectbox("Filter by Company Employee Scale:", all_sizes)

# Dynamically apply filters to data state engine out of the box
filtered_df = df_clean.copy()
if selected_country != "All Countries":
    filtered_df = filtered_df[filtered_df['Country'] == selected_country]
if selected_size != "All Sizes":
    filtered_df = filtered_df[filtered_df['no_employees'] == selected_size]

# ==========================================
# 4. APP VIEWS RENDERING ROUTINES
# ==========================================

# VIEW A: PROJECT OVERVIEW PAGE LAYOUT
if app_mode == "Project Overview":
    st.title("📊 Mental Health in Tech Workplace Analysis")
    st.subheader("Exploratory Data Analysis & Strategic Insights Dashboard")
    
    st.markdown("""
    ### Project Background Scope & Business Context
    This production analytics interface provides empirical tracking into attitudes, benefit visibility, and behavioral treatment patterns regarding **mental health across global technical engineering departments**. Utilizing raw responses from the industry benchmark study, this tool helps organizational clients quantify benefit gaps, identify operational risk vectors, and construct supportive engineering cultures.
    
    Use the sidebar selections to navigate views or filter demographic data states in real-time.
    """)
    
    st.markdown("---")
    st.subheader("💡 Global Tech Baseline Indicators")
    
    # Calculate baseline mathematical metric indicators to place into KPI display cards
    total_responses = len(filtered_df)
    
    if total_responses > 0:
        treatment_pct = (filtered_df['treatment'].value_counts(normalize=True).get('Yes', 0)) * 100
        family_hist_pct = (filtered_df['family_history'].value_counts(normalize=True).get('Yes', 0)) * 100
        remote_pct = (filtered_df['remote_work'].value_counts(normalize=True).get('Yes', 0)) * 100
        
        # Render high-impact real-time functional card structures
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        kpi_col1.metric(label="Total Survey Records Count", value=f"{total_responses:,}")
        kpi_col2.metric(label="Sought Treatment Rate", value=f"{treatment_pct:.1f}%")
        kpi_col3.metric(label="Family Background Incidences", value=f"{family_hist_pct:.1f}%")
        kpi_col4.metric(label="Distributed Remote Workforce Split", value=f"{remote_pct:.1f}%")
    else:
        st.warning("No records found matching current parameter constraints. Try widening your sidebar choices.")

    st.markdown("---")
    st.subheader("🔍 Sample Workspace Raw Data View")
    st.dataframe(filtered_df.head(10), use_container_width=True)


# VIEW B: INTERACTIVE WORKFORCE INSIGHTS DASHBOARD
elif app_mode == "Interactive Workforce Insights":
    st.title("📈 Interactive Workforce Analytics Dashboard")
    st.subheader(f"Displaying results filtered for: {selected_country} | Size Scale: {selected_size}")
    
    if len(filtered_df) == 0:
        st.warning("⚠️ No data available matching the selected filter options. Please adjust your selections in the sidebar.")
        st.stop()
        
    # SECTION 1: DEMOGRAPHICS GRAPH PLOT BLOCKS
    st.markdown("### 1. Demographic Risk Parameters Breakdown")
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.markdown("**Cleaned Workforce Age Curve Density Split**")
        fig, ax = plt.subplots()
        sns.histplot(data=filtered_df, x='Age', kde=True, bins=15, color='skyblue', ax=ax)
        ax.set_ylabel("Respondent Concentration Density")
        st.pyplot(fig)
        
    with row1_col2:
        st.markdown("**Standardised Gender Distribution Rates vs. Treatment Seeking behavior**")
        fig, ax = plt.subplots()
        sns.countplot(data=filtered_df, x='Gender', hue='treatment', palette='muted', ax=ax)
        st.pyplot(fig)
        
    st.markdown("---")
    
    # SECTION 2: CORPORATE SYSTEMS GRAPH PLOT BLOCKS
    st.markdown("### 2. Operational Frameworks & Benefits Visibility Evaluation Matrix")
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        st.markdown("**Perceived Healthcare Benefits Provisioning Matrix across corporate Sizing Brackets**")
        fig, ax = plt.subplots()
        sns.countplot(
            data=filtered_df, 
            x='no_employees', 
            hue='benefits', 
            order=['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000'], 
            palette='magma', 
            ax=ax
        )
        plt.xticks(rotation=30)
        st.pyplot(fig)
        
    with row2_col2:
        st.markdown("**Willingness to Discuss Mental Health Concerns openly with Direct Managers**")
        fig, ax = plt.subplots()
        sns.countplot(data=filtered_df, x='supervisor', palette='deep', ax=ax)
        st.pyplot(fig)

    st.markdown("---")
    
    # SECTION 3: SYSTEMIC WORK INTERFERENCE AND DISCLOSURE THRESHOLDS (COMPLETED)
    st.markdown("### 3. Work Performance Interference Factors vs. Psychological Privacy Settings")
    row3_col1, row3_col2 = st.columns(2)
    
    with row3_col1:
        st.markdown("**Work Arrangements vs. Experienced Performance Obstacles Profile**")
        fig, ax = plt.subplots()
        sns.countplot(data=filtered_df, x='remote_work', hue='work_interfere', palette='cubehelix', ax=ax)
        st.pyplot(fig)
        
    with row3_col2:
        st.markdown("**Anonymity Protection Assurance Comfort Scales across Organization Scales**")
        fig, ax = plt.subplots()
        sns.countplot(
            data=filtered_df, 
            x='no_employees', 
            hue='anonymity', 
            order=['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000'],
            ax=ax
        )
        plt.xticks(rotation=30)
        st.pyplot(fig)
        
    # SECTION 4: MULTIVARIATE CORRELATION MATRIX GRID ANALYSIS
    st.markdown("---")
    st.markdown("### 4. Multivariate Association Matrix Heatmap")
    
    # Map attributes to integer categories for generating dynamic heatmaps
    multivariate_df = filtered_df[['Age', 'Gender', 'family_history', 'treatment', 'remote_work']].copy()
    for col in ['Gender', 'family_history', 'treatment', 'remote_work']:
        multivariate_df[col] = multivariate_df[col].astype('category').cat.codes
        
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(multivariate_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)
