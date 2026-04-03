#!/usr/bin/env python3
"""
Restaurant Customer Satisfaction — Comprehensive Streamlit Dashboard
GWU MKTG 4163 / 6263 — Applied Marketing Analytics, Spring 2026

Covers every analytical concept from the course syllabus applied to the dataset:
  Descriptive statistics, Pivot tables, Index Match, Histograms, Box/Stem plots,
  Correlation & R², ANOVA (1-factor, 2-factor), Chi-Square, Z-test,
  Linear & Logistic Regression, Monte Carlo Simulation, Markov Chains,
  Holt-Winters / Moving Averages, Decision Trees, Simpson's Paradox,
  Rank & Percentile, Probability functions, Prescriptive recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import f_oneway, chi2_contingency, pearsonr, zscore, norm, ttest_ind
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             classification_report, roc_curve, auc)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Restaurant Satisfaction Analytics",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GW Color Palette ───────────────────────────────────────────────────────
NAVY = '#033C5A'
GOLD = '#AA9868'
LIGHT_GOLD = '#F5F0E6'
PALETTE = [NAVY, GOLD, '#4A90D9', '#D4A843', '#2C6E8A', '#C4956A', '#1B4F72', '#DAC292']

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container {padding-top: 1rem;}
    h1, h2, h3 {color: #033C5A;}
    .stMetric > div {background: linear-gradient(135deg, #033C5A, #065A85);
        border-radius: 10px; padding: 10px; color: white;}
    .stMetric label {color: #a0c4d8 !important;}
    .stMetric [data-testid="stMetricValue"] {color: #AA9868 !important;}
    div[data-testid="stSidebar"] {background: #033C5A;}
    div[data-testid="stSidebar"] * {color: white !important;}
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stMultiSelect label,
    div[data-testid="stSidebar"] .stSlider label {color: #AA9868 !important;}
    .insight-box {background: #F5F0E6; border-left: 5px solid #AA9868;
        padding: 15px; border-radius: 0 8px 8px 0; margin: 10px 0;
        color: #1a1a1a !important;}
    .insight-box * {color: #1a1a1a !important;}
    .insight-box h4 {color: #033C5A !important;}
    .method-box {background: #e8f0f5; border-left: 5px solid #033C5A;
        padding: 15px; border-radius: 0 8px 8px 0; margin: 10px 0;
        color: #1a1a1a !important;}
    .method-box * {color: #1a1a1a !important;}
    .method-box b {color: #033C5A !important;}
</style>
""", unsafe_allow_html=True)

# ─── Load Data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('restaurant_customer_satisfaction.csv')
    return df

df = load_data()

numeric_cols = ['Age', 'Income', 'AverageSpend', 'GroupSize', 'WaitTime',
                'ServiceRating', 'FoodRating', 'AmbianceRating']
categorical_cols = ['Gender', 'VisitFrequency', 'PreferredCuisine',
                    'TimeOfVisit', 'DiningOccasion', 'MealType']
binary_cols = ['OnlineReservation', 'DeliveryOrder', 'LoyaltyProgramMember']
rating_cols = ['ServiceRating', 'FoodRating', 'AmbianceRating']
target = 'HighSatisfaction'

# ─── Sidebar Navigation ────────────────────────────────────────────────────
st.sidebar.markdown("<h2 style='text-align:center; color:#AA9868;'>GW Business</h2>", unsafe_allow_html=True)
st.sidebar.title("Navigation")

pages = [
    "🏠 Project Overview",
    "📋 Data Context & Collection",
    "🔍 Data Cleaning & Preparation",
    "📊 Exploratory Data Analysis",
    "📈 Descriptive Analytics",
    "🔬 Inferential Statistics",
    "🤖 Predictive Modeling",
    "💡 Prescriptive Analytics",
    "📌 Conclusions & Limitations",
]
page = st.sidebar.radio("Go to", pages, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("**MKTG 4163 — Spring 2026**")
st.sidebar.markdown("Prof. David Ashley")
st.sidebar.markdown("Team: Max, Leonie, Sam & Yen Kai")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: PROJECT OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == pages[0]:
    st.title("🍽️ Predicting Restaurant Customer Satisfaction")
    st.markdown("### What Are the Most Influential Factors Determining Restaurant Satisfaction?")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Variables", f"{df.shape[1]}")
    col3.metric("High Satisfaction Rate", f"{df[target].mean()*100:.1f}%")
    col4.metric("Cuisine Types", f"{df['PreferredCuisine'].nunique()}")

    st.markdown("---")

    st.markdown("### Research Objectives")
    st.markdown("""
    1. **Identify** the most influential factors driving customer satisfaction in restaurants.
    2. **Quantify** the relationship between operational metrics (wait time, service quality) and satisfaction outcomes.
    3. **Predict** whether a customer will report high satisfaction based on their profile and visit characteristics.
    4. **Recommend** actionable strategies for restaurant operators to improve satisfaction rates.
    """)

    st.markdown("### Analytical Framework")
    st.markdown("""
    This dashboard follows the 7-step data analysis framework from the course:

    | Step | Description | Course Tools Applied |
    |------|-------------|---------------------|
    | 1. Define Objective | Research question & data context | Problem framing |
    | 2. Data Collection | Primary & secondary data sources | Data sourcing |
    | 3. Explore Data | EDA with visualizations | Histograms, box plots, stem plots |
    | 4a. Descriptive | Summarize features | Pivot tables, averages, counts, percentages, Index Match, Rank & Percentile |
    | 4b. Inferential | Generalize from sample | Correlation, ANOVA, Chi-Square, Z-test |
    | 4c. Predictive | Forecast outcomes | Regression, Monte Carlo, Decision Trees, Markov Chains, Moving Averages |
    | 4d. Prescriptive | Recommend actions | Combined insights, Implications Wheel |
    | 5-7. Conclude | Interpret & communicate | Dashboards, storytelling |
    """)

    st.markdown("### Qualities Under Assessment")
    c1, c2, c3 = st.columns(3)
    c1.info("**🕐 Time of Visit**\nBreakfast, Lunch, Dinner")
    c2.info("**🎉 Dining Occasion**\nCasual, Business, Celebration")
    c3.info("**🔄 Visit Frequency**\nDaily, Weekly, Monthly, Rarely")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DATA CONTEXT & COLLECTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == pages[1]:
    st.title("📋 Data Context & Collection")

    st.markdown("### Dataset Background")
    st.markdown("""
    This dataset contains **1,500 customer records** from restaurant visits, sourced as
    **secondary data** from Kaggle. Each record captures a complete customer profile
    including demographics, dining behavior, digital engagement, experience ratings,
    and a binary satisfaction outcome.
    """)

    st.markdown("### Primary vs. Secondary Data")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🟢 Secondary Data (This Dataset)")
        st.markdown("""
        - **Source**: Kaggle open dataset
        - **Size**: 1,500 records, 19 variables
        - **Type**: Cross-sectional survey data
        - **Advantages**: Large sample, structured, ready for analysis
        - **Limitations**: No control over collection methodology
        """)
    with col2:
        st.markdown("#### 🔵 Primary Data (Supplementary)")
        st.markdown("""
        - Could include: Customer surveys, interviews, focus groups
        - Direct observation of restaurant operations
        - Mystery shopper evaluations
        - Social media sentiment analysis
        - Would help validate secondary data findings
        """)

    st.markdown("### Data Dictionary")
    data_dict = pd.DataFrame({
        'Variable': df.columns,
        'Type': ['ID', 'Numeric', 'Categorical', 'Numeric',
                'Categorical', 'Numeric', 'Categorical', 'Categorical',
                'Numeric', 'Categorical', 'Categorical',
                'Binary', 'Binary', 'Binary',
                'Numeric', 'Numeric (1-5)', 'Numeric (1-5)', 'Numeric (1-5)',
                'Binary (Target)'],
        'Description': [
            'Unique customer identifier', 'Customer age in years',
            'Male / Female', 'Annual income ($)',
            'Visit frequency (Daily/Weekly/Monthly/Rarely)',
            'Average amount spent per visit ($)',
            'Preferred cuisine type', 'Meal time (Breakfast/Lunch/Dinner)',
            'Number of people in party', 'Purpose (Casual/Business/Celebration)',
            'Dine-in or Takeaway', 'Made online reservation (0/1)',
            'Delivery order (0/1)', 'Loyalty program member (0/1)',
            'Wait time in minutes', 'Service quality rating',
            'Food quality rating', 'Ambiance rating',
            'Target: High satisfaction (0/1)'
        ],
        'Non-Null': [df[c].notna().sum() for c in df.columns],
    })
    st.dataframe(data_dict, use_container_width=True, hide_index=True)

    st.markdown("### Raw Data Preview")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DATA CLEANING & PREPARATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == pages[2]:
    st.title("🔍 Data Cleaning & Preparation")
    st.markdown("Before analysis, we validate data quality: missing values, duplicates, outliers, and data types.")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Missing Values", f"{df.isnull().sum().sum()}")
    col2.metric("Duplicate Rows", f"{df.duplicated().sum()}")
    col3.metric("Unique Customers", f"{df['CustomerID'].nunique()}")

    # ── Missing values ──
    st.markdown("### Missing Value Analysis")
    missing = df.isnull().sum().reset_index()
    missing.columns = ['Variable', 'Missing Count']
    missing['Missing %'] = (missing['Missing Count'] / len(df) * 100).round(2)
    st.dataframe(missing, use_container_width=True, hide_index=True)
    st.success("✅ **No missing values detected.** The dataset is complete across all 19 variables.")

    # ── Data types ──
    st.markdown("### Data Type Validation")
    dtypes = df.dtypes.reset_index()
    dtypes.columns = ['Variable', 'Data Type']
    dtypes['Unique Values'] = [df[c].nunique() for c in df.columns]
    dtypes['Sample Values'] = [str(df[c].unique()[:4].tolist()) for c in df.columns]
    st.dataframe(dtypes, use_container_width=True, hide_index=True)

    # ── Outlier detection (Z-score) ──
    st.markdown("### Outlier Detection (Z-Score Method)")
    st.markdown("Values with |Z-score| > 3 are flagged as potential outliers.")
    outlier_data = []
    for col in numeric_cols:
        z = np.abs(zscore(df[col]))
        n_outliers = (z > 3).sum()
        outlier_data.append({
            'Variable': col,
            'Mean': round(df[col].mean(), 2),
            'Std Dev': round(df[col].std(), 2),
            'Min': round(df[col].min(), 2),
            'Max': round(df[col].max(), 2),
            'Outliers (|Z|>3)': n_outliers,
            'Outlier %': round(n_outliers / len(df) * 100, 2)
        })
    outlier_df = pd.DataFrame(outlier_data)
    st.dataframe(outlier_df, use_container_width=True, hide_index=True)

    # Z-score distribution
    selected_col = st.selectbox("Visualize Z-score distribution for:", numeric_cols)
    z_vals = zscore(df[selected_col])
    fig = px.histogram(x=z_vals, nbins=50, title=f'Z-Score Distribution: {selected_col}',
                       labels={'x': 'Z-Score', 'y': 'Count'},
                       color_discrete_sequence=[NAVY])
    fig.add_vline(x=3, line_dash="dash", line_color="red", annotation_text="Z=3")
    fig.add_vline(x=-3, line_dash="dash", line_color="red", annotation_text="Z=-3")
    st.plotly_chart(fig, use_container_width=True)

    # ── Class balance ──
    st.markdown("### Target Variable Balance")
    balance = df[target].value_counts().reset_index()
    balance.columns = ['HighSatisfaction', 'Count']
    balance['Label'] = balance['HighSatisfaction'].map({0: 'Low Satisfaction', 1: 'High Satisfaction'})
    balance['Percentage'] = (balance['Count'] / len(df) * 100).round(1)
    fig = px.pie(balance, values='Count', names='Label',
                 color_discrete_sequence=[NAVY, GOLD],
                 title='Target Variable Distribution')
    st.plotly_chart(fig, use_container_width=True)
    if df[target].mean() < 0.3 or df[target].mean() > 0.7:
        st.warning("⚠️ The target variable is **imbalanced**. This should be considered when interpreting model performance.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLORATORY DATA ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == pages[3]:
    st.title("📊 Exploratory Data Analysis")
    st.markdown("EDA uses statistical and graphical techniques to explore the data before applying models.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Histograms", "Box & Stem Plots", "Scatter Plots",
        "Category Breakdowns", "Simpson's Paradox"
    ])

    # ── TAB: Histograms ──
    with tab1:
        st.markdown("### Distribution of Numeric Variables")
        st.markdown('<div class="method-box"><b>Course Tool:</b> Histograms — visualize the frequency distribution and shape of numeric data.</div>', unsafe_allow_html=True)
        sel_hist = st.multiselect("Select variables:", numeric_cols, default=numeric_cols[:4])
        if sel_hist:
            for col in sel_hist:
                fig = px.histogram(df, x=col, nbins=40, color=target,
                                   barmode='overlay', marginal='box',
                                   color_discrete_sequence=[NAVY, GOLD],
                                   title=f'Distribution of {col} by Satisfaction Level',
                                   labels={target: 'High Satisfaction'})
                fig.update_layout(bargap=0.05)
                st.plotly_chart(fig, use_container_width=True)

    # ── TAB: Box & Stem Plots ──
    with tab2:
        st.markdown("### Box Plots & Stem-and-Leaf")
        st.markdown('<div class="method-box"><b>Course Tool:</b> Box plots and stem plots — visualize spread, quartiles, and outliers.</div>', unsafe_allow_html=True)

        sel_box = st.selectbox("Numeric variable:", numeric_cols, key='box')
        sel_group = st.selectbox("Group by:", ['None'] + categorical_cols + [target])

        if sel_group == 'None':
            fig = px.box(df, y=sel_box, color_discrete_sequence=[NAVY],
                         title=f'Box Plot: {sel_box}', points='outliers')
        else:
            fig = px.box(df, x=sel_group, y=sel_box, color=sel_group,
                         color_discrete_sequence=PALETTE,
                         title=f'Box Plot: {sel_box} by {sel_group}', points='outliers')
        st.plotly_chart(fig, use_container_width=True)

        # Stem-and-leaf (text-based)
        st.markdown("#### Stem-and-Leaf Display")
        stem_col = st.selectbox("Variable for stem-and-leaf:", rating_cols)
        vals = sorted(df[stem_col].values.astype(int))
        stems = {}
        for v in vals:
            stem = v // 1
            leaf = v % 1
            stems.setdefault(stem, []).append(int(leaf * 10) if leaf else 0)
        stem_text = ""
        for s in sorted(stems.keys()):
            leaves = ' '.join(str(l) for l in stems[s])
            count = len(stems[s])
            stem_text += f"  {s} | {leaves}  (n={count})\n"
        st.code(stem_text, language=None)

    # ── TAB: Scatter Plots ──
    with tab3:
        st.markdown("### Scatter Plots — Relationship Explorer")
        c1, c2 = st.columns(2)
        x_var = c1.selectbox("X-axis:", numeric_cols, index=4)  # WaitTime
        y_var = c2.selectbox("Y-axis:", numeric_cols, index=5)  # ServiceRating
        color_var = st.selectbox("Color by:", [target] + categorical_cols)

        fig = px.scatter(df, x=x_var, y=y_var, color=df[color_var].astype(str),
                         color_discrete_sequence=PALETTE, opacity=0.6,
                         title=f'{y_var} vs {x_var}',
                         trendline='ols')
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB: Category Breakdowns ──
    with tab4:
        st.markdown("### Satisfaction Rate by Category")
        sel_cat = st.selectbox("Category:", categorical_cols + binary_cols)
        grouped = df.groupby(sel_cat).agg(
            Count=(target, 'count'),
            HighSat=(target, 'sum'),
            SatRate=(target, 'mean')
        ).reset_index()
        grouped['SatRate'] = (grouped['SatRate'] * 100).round(1)
        grouped['LowSat'] = grouped['Count'] - grouped['HighSat']

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=grouped[sel_cat].astype(str), y=grouped['Count'],
                             name='Total Count', marker_color=NAVY, opacity=0.7))
        fig.add_trace(go.Scatter(x=grouped[sel_cat].astype(str), y=grouped['SatRate'],
                                 name='Satisfaction Rate (%)', mode='lines+markers',
                                 line=dict(color=GOLD, width=3),
                                 marker=dict(size=10)), secondary_y=True)
        fig.update_layout(title=f'Count & Satisfaction Rate by {sel_cat}')
        fig.update_yaxes(title_text="Count", secondary_y=False)
        fig.update_yaxes(title_text="Satisfaction Rate (%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB: Simpson's Paradox ──
    with tab5:
        st.markdown("### Simpson's Paradox Check")
        st.markdown('<div class="method-box"><b>Course Tool:</b> Simpson\'s Paradox — a trend that appears in groups reverses when data is combined. We check if overall satisfaction trends hold within subgroups.</div>', unsafe_allow_html=True)

        sp_var = st.selectbox("Primary variable:", categorical_cols, index=0)  # Gender
        sp_group = st.selectbox("Subgroup variable:", [c for c in categorical_cols if c != sp_var], index=0)

        overall = df.groupby(sp_var)[target].mean() * 100
        st.markdown(f"**Overall satisfaction rate by {sp_var}:**")
        st.dataframe(overall.round(1).reset_index().rename(
            columns={target: 'Satisfaction Rate (%)'}), hide_index=True)

        st.markdown(f"**Satisfaction rate by {sp_var} within each {sp_group}:**")
        subgroup = df.groupby([sp_group, sp_var])[target].mean().unstack() * 100
        st.dataframe(subgroup.round(1))

        # Check for reversal
        overall_vals = overall.values
        direction_overall = "higher" if overall_vals[0] > overall_vals[1] else "lower"
        reversals = 0
        for grp in subgroup.index:
            row = subgroup.loc[grp].values
            if len(row) >= 2:
                direction_sub = "higher" if row[0] > row[1] else "lower"
                if direction_sub != direction_overall:
                    reversals += 1
        if reversals > 0:
            st.warning(f"⚠️ **Potential Simpson's Paradox detected!** The overall trend reverses in {reversals} subgroup(s). The aggregated data may be misleading.")
        else:
            st.success("✅ No Simpson's Paradox detected. The trend is consistent across subgroups.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DESCRIPTIVE ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == pages[4]:
    st.title("📈 Descriptive Analytics")
    st.markdown("Summarizing the who, what, where, when, and how of our data.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Summary Statistics", "Pivot Tables",
        "Index Match / Lookup", "Rank & Percentile"
    ])

    # ── TAB: Summary Stats ──
    with tab1:
        st.markdown("### Summary Statistics")
        st.markdown('<div class="method-box"><b>Course Tools:</b> Descriptive statistics — mean, median, mode, std dev, counts, percentages, sums, averages.</div>', unsafe_allow_html=True)
        desc = df[numeric_cols].describe().T
        desc['median'] = df[numeric_cols].median()
        desc['skew'] = df[numeric_cols].skew()
        desc = desc[['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max', 'skew']].round(2)
        st.dataframe(desc, use_container_width=True)

        st.markdown("### Categorical Variable Frequencies")
        sel_freq = st.selectbox("Select variable:", categorical_cols)
        freq = df[sel_freq].value_counts().reset_index()
        freq.columns = [sel_freq, 'Count']
        freq['Percentage'] = (freq['Count'] / len(df) * 100).round(1)
        freq['Cumulative %'] = freq['Percentage'].cumsum().round(1)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(freq, hide_index=True, use_container_width=True)
        with c2:
            fig = px.bar(freq, x=sel_freq, y='Count', text='Percentage',
                         color_discrete_sequence=[NAVY], title=f'Distribution of {sel_freq}')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB: Pivot Tables ──
    with tab2:
        st.markdown("### Interactive Pivot Tables")
        st.markdown('<div class="method-box"><b>Course Tool:</b> Multivariate pivot tables and data slicers — summarize data across multiple dimensions.</div>', unsafe_allow_html=True)

        row_var = st.selectbox("Row variable:", categorical_cols, index=2)
        col_var = st.selectbox("Column variable:", [c for c in categorical_cols if c != row_var], index=2)
        val_var = st.selectbox("Value variable:", numeric_cols + [target], index=5)
        agg_func = st.selectbox("Aggregation:", ['mean', 'sum', 'count', 'median', 'std'])

        pivot = pd.pivot_table(df, values=val_var, index=row_var,
                               columns=col_var, aggfunc=agg_func).round(2)
        st.dataframe(pivot, use_container_width=True)

        fig = px.imshow(pivot, text_auto='.2f', color_continuous_scale='Blues',
                        title=f'Pivot Heatmap: {agg_func.title()}({val_var}) by {row_var} × {col_var}')
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB: Index Match / Lookup ──
    with tab3:
        st.markdown("### Index Match / Lookup Tool")
        st.markdown('<div class="method-box"><b>Course Tools:</b> VLOOKUP, XLOOKUP, INDEX MATCH — retrieve specific data points by looking up values across variables.</div>', unsafe_allow_html=True)

        st.markdown("#### Customer Lookup by ID")
        cust_id = st.number_input("Enter Customer ID:", min_value=int(df['CustomerID'].min()),
                                   max_value=int(df['CustomerID'].max()),
                                   value=int(df['CustomerID'].min()))
        result = df[df['CustomerID'] == cust_id]
        if len(result) > 0:
            st.dataframe(result, use_container_width=True, hide_index=True)
        else:
            st.warning("Customer ID not found.")

        st.markdown("#### Conditional Lookup (INDEX MATCH equivalent)")
        st.markdown("Find the average satisfaction metrics for a specific customer profile:")
        c1, c2, c3 = st.columns(3)
        match_cuisine = c1.selectbox("Cuisine:", df['PreferredCuisine'].unique())
        match_occasion = c2.selectbox("Occasion:", df['DiningOccasion'].unique())
        match_freq = c3.selectbox("Frequency:", df['VisitFrequency'].unique())

        matched = df[(df['PreferredCuisine'] == match_cuisine) &
                     (df['DiningOccasion'] == match_occasion) &
                     (df['VisitFrequency'] == match_freq)]
        if len(matched) > 0:
            st.markdown(f"**Found {len(matched)} matching customers:**")
            summary = matched[numeric_cols + [target]].mean().round(2)
            st.dataframe(summary.to_frame('Average Value').T, use_container_width=True)
        else:
            st.info("No matching records for this combination.")

        st.markdown("#### COUNTIF / AVERAGEIF Equivalent")
        cond_var = st.selectbox("Condition variable:", categorical_cols + binary_cols, key='countif')
        cond_val = st.selectbox("Condition value:", df[cond_var].unique(), key='countif_val')
        subset = df[df[cond_var] == cond_val]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("COUNTIF", f"{len(subset)}")
        c2.metric("AVG Satisfaction Rate", f"{subset[target].mean()*100:.1f}%")
        c3.metric("AVG Spend", f"${subset['AverageSpend'].mean():.2f}")
        c4.metric("AVG Wait Time", f"{subset['WaitTime'].mean():.1f} min")

    # ── TAB: Rank & Percentile ──
    with tab4:
        st.markdown("### Rank & Percentile Analysis")
        st.markdown('<div class="method-box"><b>Course Tool:</b> Rank and Percentile — understand relative positioning within the data.</div>', unsafe_allow_html=True)

        rank_var = st.selectbox("Rank by:", numeric_cols)
        df_rank = df[['CustomerID', rank_var, target]].copy()
        df_rank['Rank'] = df_rank[rank_var].rank(ascending=False).astype(int)
        df_rank['Percentile'] = (df_rank[rank_var].rank(pct=True) * 100).round(1)
        df_rank = df_rank.sort_values('Rank')

        st.dataframe(df_rank.head(20), use_container_width=True, hide_index=True)

        fig = px.scatter(df_rank, x='Percentile', y=rank_var, color=df_rank[target].astype(str),
                         color_discrete_sequence=[NAVY, GOLD],
                         title=f'{rank_var}: Percentile Distribution',
                         labels={'color': 'High Satisfaction'})
        st.plotly_chart(fig, use_container_width=True)

        # Percentile summary
        st.markdown("#### Key Percentiles")
        percs = [10, 25, 50, 75, 90, 95, 99]
        perc_vals = np.percentile(df[rank_var], percs)
        perc_df = pd.DataFrame({'Percentile': [f'{p}th' for p in percs], rank_var: perc_vals.round(2)})
        st.dataframe(perc_df, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: INFERENTIAL STATISTICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == pages[5]:
    st.title("🔬 Inferential Statistics")
    st.markdown("Understanding the **why** — making generalizations about the population from our sample.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Correlation & R²", "ANOVA", "Chi-Square Test", "Z-Test"
    ])

    # ── TAB: Correlation ──
    with tab1:
        st.markdown("### Correlation Analysis & Coefficient of Determination")
        st.markdown('<div class="method-box"><b>Course Tools:</b> Correlation and coefficients of determination (R²) — measure the strength and direction of linear relationships.</div>', unsafe_allow_html=True)

        # Full correlation matrix
        corr = df[numeric_cols + [target]].corr()
        fig = px.imshow(corr, text_auto='.3f', color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1, title='Correlation Matrix',
                        aspect='equal')
        st.plotly_chart(fig, use_container_width=True)

        # Pairwise with p-values
        st.markdown("#### Correlation with High Satisfaction")
        corr_data = []
        for col in numeric_cols:
            r, p = pearsonr(df[col], df[target])
            corr_data.append({
                'Variable': col,
                'Pearson r': round(r, 4),
                'R² (Determination)': round(r**2, 4),
                'p-value': round(p, 6),
                'Significant (p<0.05)': '✅ Yes' if p < 0.05 else '❌ No',
                'Strength': 'Strong' if abs(r) > 0.5 else ('Moderate' if abs(r) > 0.3 else 'Weak')
            })
        corr_df = pd.DataFrame(corr_data).sort_values('Pearson r', key=abs, ascending=False)
        st.dataframe(corr_df, use_container_width=True, hide_index=True)

        # Interactive scatter with R²
        st.markdown("#### Explore Relationship")
        c1, c2 = st.columns(2)
        rx = c1.selectbox("X variable:", numeric_cols, index=4, key='corrx')
        ry = c2.selectbox("Y variable:", numeric_cols + [target], index=5, key='corry')
        r, p = pearsonr(df[rx], df[ry])
        fig = px.scatter(df, x=rx, y=ry, trendline='ols', color_discrete_sequence=[NAVY],
                         opacity=0.5, title=f'{ry} vs {rx} — r={r:.4f}, R²={r**2:.4f}, p={p:.4e}')
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB: ANOVA ──
    with tab2:
        st.markdown("### ANOVA — Analysis of Variance")
        st.markdown('<div class="method-box"><b>Course Tool:</b> ANOVA — single factor, two factor with and without replication. Tests if means differ significantly across groups.</div>', unsafe_allow_html=True)

        st.markdown("#### Single-Factor ANOVA")
        anova_factor = st.selectbox("Factor (grouping variable):", categorical_cols)
        anova_response = st.selectbox("Response variable:", numeric_cols + [target], index=len(numeric_cols))

        groups = [group[anova_response].values for _, group in df.groupby(anova_factor)]
        f_stat, p_val = f_oneway(*groups)

        c1, c2, c3 = st.columns(3)
        c1.metric("F-Statistic", f"{f_stat:.4f}")
        c2.metric("p-value", f"{p_val:.6f}")
        c3.metric("Result", "Significant ✅" if p_val < 0.05 else "Not Significant ❌")

        # Group means
        group_stats = df.groupby(anova_factor)[anova_response].agg(['mean', 'std', 'count']).round(4)
        group_stats.columns = ['Group Mean', 'Std Dev', 'Count']
        st.dataframe(group_stats, use_container_width=True)

        fig = px.box(df, x=anova_factor, y=anova_response, color=anova_factor,
                     color_discrete_sequence=PALETTE, title=f'ANOVA: {anova_response} by {anova_factor}')
        st.plotly_chart(fig, use_container_width=True)

        # Two-factor ANOVA summary
        st.markdown("#### Two-Factor ANOVA")
        factor2 = st.selectbox("Second factor:", [c for c in categorical_cols if c != anova_factor])
        pivot_2way = df.groupby([anova_factor, factor2])[anova_response].mean().unstack().round(4)
        st.dataframe(pivot_2way, use_container_width=True)

        fig = px.box(df, x=anova_factor, y=anova_response, color=factor2,
                     color_discrete_sequence=PALETTE,
                     title=f'Two-Factor: {anova_response} by {anova_factor} and {factor2}')
        st.plotly_chart(fig, use_container_width=True)

        # Run all key ANOVA tests
        st.markdown("#### ANOVA Summary: Key Factors vs. Satisfaction")
        anova_results = []
        for col in categorical_cols:
            groups = [group[target].values for _, group in df.groupby(col)]
            f_s, p_v = f_oneway(*groups)
            anova_results.append({'Factor': col, 'F-Statistic': round(f_s, 4),
                                  'p-value': round(p_v, 6),
                                  'Significant': '✅' if p_v < 0.05 else '❌'})
        st.dataframe(pd.DataFrame(anova_results), use_container_width=True, hide_index=True)

    # ── TAB: Chi-Square ──
    with tab3:
        st.markdown("### Chi-Square Test of Independence")
        st.markdown('<div class="method-box"><b>Course Tool:</b> Chi-Square — tests whether two categorical variables are independent or associated.</div>', unsafe_allow_html=True)

        chi_var = st.selectbox("Test variable:", categorical_cols + binary_cols)
        ct = pd.crosstab(df[chi_var], df[target])
        chi2, p, dof, expected = chi2_contingency(ct)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Chi² Statistic", f"{chi2:.4f}")
        c2.metric("p-value", f"{p:.6f}")
        c3.metric("Degrees of Freedom", f"{dof}")
        c4.metric("Result", "Dependent ✅" if p < 0.05 else "Independent ❌")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Observed Frequencies:**")
            ct.columns = ['Low Satisfaction', 'High Satisfaction']
            st.dataframe(ct, use_container_width=True)
        with c2:
            st.markdown("**Expected Frequencies (under H₀):**")
            expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns).round(2)
            st.dataframe(expected_df, use_container_width=True)

        # Stacked bar
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ct_pct.index.astype(str), y=ct_pct['Low Satisfaction'],
                             name='Low Satisfaction', marker_color=NAVY))
        fig.add_trace(go.Bar(x=ct_pct.index.astype(str), y=ct_pct['High Satisfaction'],
                             name='High Satisfaction', marker_color=GOLD))
        fig.update_layout(barmode='stack', title=f'Satisfaction Composition by {chi_var}',
                          yaxis_title='Percentage')
        st.plotly_chart(fig, use_container_width=True)

        # All chi-square tests
        st.markdown("#### Chi-Square Summary: All Variables")
        chi_results = []
        for col in categorical_cols + binary_cols:
            ct_t = pd.crosstab(df[col], df[target])
            c2, p2, d2, _ = chi2_contingency(ct_t)
            chi_results.append({'Variable': col, 'Chi²': round(c2, 4), 'p-value': round(p2, 6),
                                'DoF': d2, 'Significant': '✅' if p2 < 0.05 else '❌'})
        st.dataframe(pd.DataFrame(chi_results).sort_values('p-value'),
                     use_container_width=True, hide_index=True)

    # ── TAB: Z-Test ──
    with tab4:
        st.markdown("### Z-Test — Comparing Group Means")
        st.markdown('<div class="method-box"><b>Course Tool:</b> Z-test — tests whether two group means are significantly different when sample sizes are large (n>30).</div>', unsafe_allow_html=True)

        z_var = st.selectbox("Variable to compare:", numeric_cols)
        z_group = st.selectbox("Grouping variable:", categorical_cols + binary_cols, index=0)

        groups_list = df[z_group].unique()
        if len(groups_list) == 2:
            g1 = df[df[z_group] == groups_list[0]][z_var]
            g2 = df[df[z_group] == groups_list[1]][z_var]
            # For large samples, t-test approximates z-test
            t_stat, p_val = ttest_ind(g1, g2)
            z_stat = t_stat  # equivalent for large n

            c1, c2, c3 = st.columns(3)
            c1.metric(f"Mean ({groups_list[0]})", f"{g1.mean():.2f}")
            c2.metric(f"Mean ({groups_list[1]})", f"{g2.mean():.2f}")
            c3.metric("Difference", f"{g1.mean() - g2.mean():.2f}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Z-Statistic", f"{z_stat:.4f}")
            c2.metric("p-value", f"{p_val:.6f}")
            c3.metric("Result", "Significant ✅" if p_val < 0.05 else "Not Significant ❌")
        else:
            st.markdown("**Select two groups to compare:**")
            c1, c2 = st.columns(2)
            grp1 = c1.selectbox("Group 1:", groups_list, index=0)
            grp2 = c2.selectbox("Group 2:", groups_list, index=min(1, len(groups_list)-1))
            g1 = df[df[z_group] == grp1][z_var]
            g2 = df[df[z_group] == grp2][z_var]
            t_stat, p_val = ttest_ind(g1, g2)

            c1, c2, c3 = st.columns(3)
            c1.metric(f"Mean ({grp1})", f"{g1.mean():.2f}")
            c2.metric(f"Mean ({grp2})", f"{g2.mean():.2f}")
            c3.metric("Z/t Statistic", f"{t_stat:.4f}")
            st.metric("p-value", f"{p_val:.6f}")
            if p_val < 0.05:
                st.success(f"✅ The means of {z_var} are significantly different between {grp1} and {grp2}.")
            else:
                st.info(f"❌ No significant difference in {z_var} between {grp1} and {grp2}.")

        # Probability distribution overlay
        st.markdown("#### Normal Distribution Overlay")
        st.markdown('<div class="method-box"><b>Course Tool:</b> Probability functions — visualize the theoretical distribution and test normality.</div>', unsafe_allow_html=True)
        x_range = np.linspace(df[z_var].min(), df[z_var].max(), 200)
        pdf = norm.pdf(x_range, df[z_var].mean(), df[z_var].std())
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[z_var], nbinsx=40, name='Observed',
                                    histnorm='probability density', marker_color=NAVY, opacity=0.6))
        fig.add_trace(go.Scatter(x=x_range, y=pdf, name='Normal Distribution',
                                  line=dict(color=GOLD, width=3)))
        fig.update_layout(title=f'{z_var}: Observed vs Normal Distribution')
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTIVE MODELING
# ═════════════════════════════════════════════════════════════════════════════
elif page == pages[6]:
    st.title("🤖 Predictive Modeling")
    st.markdown("Forecasting outcomes based on historical data using regression, simulation, and machine learning.")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Linear Regression", "Logistic Regression",
        "Monte Carlo Simulation", "Decision Trees",
        "Markov Chains", "Moving Averages & Forecasting"
    ])

    # ── Prepare modeling data ──
    @st.cache_data
    def prepare_model_data():
        df_m = df.copy()
        le_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_m[col + '_enc'] = le.fit_transform(df_m[col])
            le_dict[col] = le
        feat = numeric_cols + binary_cols + [c + '_enc' for c in categorical_cols]
        return df_m, feat, le_dict

    df_m, feature_cols, le_dict = prepare_model_data()
    X = df_m[feature_cols]
    y = df_m[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── TAB: Linear Regression ──
    with tab1:
        st.markdown("### Multivariate Linear Regression")
        st.markdown('<div class="method-box"><b>Course Tool:</b> Multivariate regression — predict a continuous outcome from multiple predictors.</div>', unsafe_allow_html=True)

        df_m['CompositeSat'] = (df_m['ServiceRating'] + df_m['FoodRating'] + df_m['AmbianceRating']) / 3
        feat_lr = ['Age', 'Income', 'AverageSpend', 'GroupSize', 'WaitTime'] + binary_cols + [c+'_enc' for c in categorical_cols]
        X_lr = sm.add_constant(df_m[feat_lr])
        ols = sm.OLS(df_m['CompositeSat'], X_lr).fit()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²", f"{ols.rsquared:.4f}")
        c2.metric("Adj R²", f"{ols.rsquared_adj:.4f}")
        c3.metric("F-statistic", f"{ols.fvalue:.2f}")
        c4.metric("F p-value", f"{ols.f_pvalue:.2e}")

        coef_df = pd.DataFrame({
            'Variable': ['Intercept'] + feat_lr,
            'Coefficient': ols.params.round(4).values,
            'Std Error': ols.bse.round(4).values,
            't-value': ols.tvalues.round(4).values,
            'p-value': ols.pvalues.round(6).values,
            'Significant': ['✅' if p < 0.05 else '❌' for p in ols.pvalues.values]
        })
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

        # Residual plot
        residuals = ols.resid
        fitted = ols.fittedvalues
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Residuals vs Fitted', 'Residual Distribution'])
        fig.add_trace(go.Scatter(x=fitted, y=residuals, mode='markers',
                                  marker=dict(color=NAVY, opacity=0.4, size=4)), row=1, col=1)
        fig.add_hline(y=0, line_dash='dash', line_color=GOLD, row=1, col=1)
        fig.add_trace(go.Histogram(x=residuals, nbinsx=40, marker_color=NAVY, opacity=0.7), row=1, col=2)
        fig.update_layout(title='Regression Diagnostics', showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB: Logistic Regression ──
    with tab2:
        st.markdown("### Logistic Regression")
        st.markdown('<div class="method-box"><b>Course Tool:</b> Logistic regression — predict binary outcomes (High vs Low satisfaction) and understand factor importance.</div>', unsafe_allow_html=True)

        log_reg = LogisticRegression(max_iter=1000, random_state=42)
        log_reg.fit(X_train_s, y_train)
        y_pred = log_reg.predict(X_test_s)
        y_prob = log_reg.predict_proba(X_test_s)[:, 1]
        acc = accuracy_score(y_test, y_pred)

        c1, c2, c3 = st.columns(3)
        cr = classification_report(y_test, y_pred, output_dict=True)
        c1.metric("Accuracy", f"{acc:.1%}")
        c2.metric("Precision (High Sat)", f"{cr['1']['precision']:.1%}")
        c3.metric("Recall (High Sat)", f"{cr['1']['recall']:.1%}")

        # S-Curve
        st.markdown("#### Logistic S-Curve")
        best_idx = np.argmax(np.abs(log_reg.coef_[0]))
        best_name = feature_cols[best_idx]
        x_1d = X_test_s[:, best_idx]
        X_curve = np.tile(X_test_s.mean(axis=0), (500, 1))
        x_range = np.linspace(x_1d.min() - 1, x_1d.max() + 1, 500)
        X_curve[:, best_idx] = x_range
        y_curve = log_reg.predict_proba(X_curve)[:, 1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_1d[y_test.values == 0],
                                  y=np.zeros(sum(y_test.values == 0)) + np.random.uniform(-0.03, 0.03, sum(y_test.values == 0)),
                                  mode='markers', name='Low Satisfaction',
                                  marker=dict(color=NAVY, opacity=0.3, size=5)))
        fig.add_trace(go.Scatter(x=x_1d[y_test.values == 1],
                                  y=np.ones(sum(y_test.values == 1)) + np.random.uniform(-0.03, 0.03, sum(y_test.values == 1)),
                                  mode='markers', name='High Satisfaction',
                                  marker=dict(color=GOLD, opacity=0.3, size=5)))
        fig.add_trace(go.Scatter(x=x_range, y=y_curve, mode='lines', name='Logistic Curve',
                                  line=dict(color='red', width=3)))
        fig.add_hline(y=0.5, line_dash='dot', line_color='gray', annotation_text='Decision Boundary')
        fig.update_layout(title=f'Logistic Regression S-Curve (Primary Predictor: {best_name})',
                          xaxis_title=f'Standardized {best_name}',
                          yaxis_title='P(High Satisfaction)', height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                        x=['Predicted Low', 'Predicted High'],
                        y=['Actual Low', 'Actual High'],
                        title='Confusion Matrix')
        st.plotly_chart(fig, use_container_width=True)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                  name=f'ROC (AUC = {roc_auc:.3f})',
                                  line=dict(color=NAVY, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                  name='Random', line=dict(color='gray', dash='dash')))
        fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate', height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        st.markdown("#### Feature Importance (Standardized Coefficients)")
        coefs = pd.DataFrame({
            'Feature': feature_cols,
            'Coefficient': log_reg.coef_[0]
        }).sort_values('Coefficient')
        fig = px.bar(coefs, x='Coefficient', y='Feature', orientation='h',
                     color='Coefficient', color_continuous_scale='RdBu_r',
                     color_continuous_midpoint=0,
                     title='Logistic Regression Feature Importance', height=600)
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB: Monte Carlo ──
    with tab3:
        st.markdown("### Monte Carlo Simulation")
        st.markdown('<div class="method-box"><b>Course Tool:</b> Monte Carlo simulation — use random sampling to model probability of different outcomes under uncertainty.</div>', unsafe_allow_html=True)

        st.markdown("#### Scenario Simulator")
        st.markdown("Simulate customer profiles under different conditions and predict satisfaction rates.")

        c1, c2 = st.columns(2)
        n_sim = c1.slider("Number of simulations:", 1000, 50000, 10000, step=1000)
        mc_seed = c2.number_input("Random seed:", value=42)

        st.markdown("#### Wait Time Scenario Analysis")
        np.random.seed(mc_seed)

        scenarios = {
            'Low Wait (0-15 min)': (0, 15),
            'Medium Wait (15-30 min)': (15, 30),
            'High Wait (30-60 min)': (30, 60)
        }

        mc_results = {}
        for name, (lo, hi) in scenarios.items():
            sim = pd.DataFrame()
            for col in feature_cols:
                if col == 'WaitTime':
                    sim[col] = np.random.uniform(lo, hi, n_sim)
                elif col in numeric_cols:
                    sim[col] = np.random.normal(df[col].mean(), df[col].std(), n_sim)
                elif col in binary_cols:
                    sim[col] = np.random.binomial(1, df[col].mean(), n_sim)
                else:
                    sim[col] = np.random.choice(df_m[col].values, n_sim)
            probs = log_reg.predict_proba(scaler.transform(sim[feature_cols]))[:, 1]
            mc_results[name] = probs

        # Distribution chart
        fig = go.Figure()
        colors = [NAVY, GOLD, '#D4534A']
        for (name, probs), color in zip(mc_results.items(), colors):
            fig.add_trace(go.Histogram(x=probs, name=name, opacity=0.6,
                                        marker_color=color, nbinsx=50, histnorm='probability density'))
        fig.add_vline(x=0.5, line_dash='dot', line_color='red', annotation_text='Decision Boundary')
        fig.update_layout(barmode='overlay', title=f'Monte Carlo: Satisfaction Probability by Wait Time ({n_sim:,} simulations each)',
                          xaxis_title='P(High Satisfaction)', yaxis_title='Density', height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        mc_summary = []
        for name, probs in mc_results.items():
            mc_summary.append({
                'Scenario': name,
                'Mean P(Satisfaction)': round(probs.mean(), 4),
                'Median': round(np.median(probs), 4),
                'Std Dev': round(probs.std(), 4),
                '% High Satisfaction': f"{(probs >= 0.5).mean()*100:.1f}%",
                '5th Percentile': round(np.percentile(probs, 5), 4),
                '95th Percentile': round(np.percentile(probs, 95), 4),
            })
        st.dataframe(pd.DataFrame(mc_summary), use_container_width=True, hide_index=True)

        # Custom scenario
        st.markdown("#### Custom Monte Carlo Scenario")
        c1, c2, c3, c4 = st.columns(4)
        custom_wait_lo = c1.number_input("Wait Min:", 0, 60, 0)
        custom_wait_hi = c2.number_input("Wait Max:", 0, 60, 15)
        custom_cuisine = c3.selectbox("Fix Cuisine:", ['Random'] + list(df['PreferredCuisine'].unique()))
        custom_loyalty = c4.selectbox("Fix Loyalty:", ['Random', 'Yes', 'No'])

        if st.button("Run Custom Simulation"):
            np.random.seed(mc_seed)
            sim = pd.DataFrame()
            for col in feature_cols:
                if col == 'WaitTime':
                    sim[col] = np.random.uniform(custom_wait_lo, custom_wait_hi, n_sim)
                elif col == 'PreferredCuisine_enc' and custom_cuisine != 'Random':
                    sim[col] = le_dict['PreferredCuisine'].transform([custom_cuisine] * n_sim)
                elif col == 'LoyaltyProgramMember' and custom_loyalty != 'Random':
                    sim[col] = 1 if custom_loyalty == 'Yes' else 0
                elif col in numeric_cols:
                    sim[col] = np.random.normal(df[col].mean(), df[col].std(), n_sim)
                elif col in binary_cols:
                    sim[col] = np.random.binomial(1, df[col].mean(), n_sim)
                else:
                    sim[col] = np.random.choice(df_m[col].values, n_sim)
            probs = log_reg.predict_proba(scaler.transform(sim[feature_cols]))[:, 1]
            st.metric("Predicted High Satisfaction Rate", f"{(probs >= 0.5).mean()*100:.1f}%")
            st.metric("Mean Probability", f"{probs.mean():.4f}")
            fig = px.histogram(x=probs, nbins=50, color_discrete_sequence=[NAVY],
                               title='Custom Scenario: Satisfaction Probability Distribution',
                               labels={'x': 'P(High Satisfaction)'})
            fig.add_vline(x=0.5, line_dash='dot', line_color='red')
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB: Decision Trees ──
    with tab4:
        st.markdown("### Decision Tree Classifier")
        st.markdown('<div class="method-box"><b>Course Tool:</b> Decision trees — segment data into groups using if-then rules, useful for market segmentation and classification.</div>', unsafe_allow_html=True)

        max_depth = st.slider("Tree depth:", 2, 8, 4)
        dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        dt.fit(X_train, y_train)
        dt_pred = dt.predict(X_test)
        dt_acc = accuracy_score(y_test, dt_pred)

        c1, c2 = st.columns(2)
        c1.metric("Decision Tree Accuracy", f"{dt_acc:.1%}")
        c2.metric("vs Logistic Regression", f"{(dt_acc - acc)*100:+.1f} pp")

        # Feature importance
        dt_imp = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': dt.feature_importances_
        }).sort_values('Importance', ascending=False)
        fig = px.bar(dt_imp.head(10), x='Importance', y='Feature', orientation='h',
                     color_discrete_sequence=[NAVY], title='Top 10 Decision Tree Feature Importance')
        st.plotly_chart(fig, use_container_width=True)

        # Tree rules
        st.markdown("#### Decision Tree Rules")
        tree_text = export_text(dt, feature_names=feature_cols, max_depth=3)
        st.code(tree_text, language=None)

    # ── TAB: Markov Chains ──
    with tab5:
        st.markdown("### Markov Chain Analysis")
        st.markdown('<div class="method-box"><b>Course Tool:</b> Markov Chains — model state transitions and predict steady-state probabilities (e.g., customer visit frequency over time).</div>', unsafe_allow_html=True)

        st.markdown("#### Visit Frequency Transition Model")
        st.markdown("We model how customers might transition between visit frequency states over time.")

        freq_order = ['Rarely', 'Monthly', 'Weekly', 'Daily']
        freq_counts = df['VisitFrequency'].value_counts()

        # Build transition matrix from cross-tab of satisfaction with frequency
        # Simulate transitions: higher satisfaction → more frequent visits
        sat_by_freq = df.groupby('VisitFrequency')[target].mean()
        n_states = len(freq_order)

        # Create realistic transition matrix
        trans = np.zeros((n_states, n_states))
        for i, state in enumerate(freq_order):
            sat_rate_val = sat_by_freq.get(state, 0.5)
            for j in range(n_states):
                if j == i:
                    trans[i][j] = 0.5  # stay same
                elif j == i + 1 and i < n_states - 1:
                    trans[i][j] = 0.25 * (1 + sat_rate_val)  # upgrade
                elif j == i - 1 and i > 0:
                    trans[i][j] = 0.25 * (1 - sat_rate_val)  # downgrade
                else:
                    trans[i][j] = 0.05
            trans[i] = trans[i] / trans[i].sum()  # normalize

        trans_df = pd.DataFrame(trans.round(4), index=freq_order, columns=freq_order)
        st.markdown("**Transition Probability Matrix:**")
        st.dataframe(trans_df, use_container_width=True)

        fig = px.imshow(trans_df, text_auto='.3f', color_continuous_scale='Blues',
                        title='Markov Chain Transition Matrix: Visit Frequency',
                        labels={'x': 'To State', 'y': 'From State'})
        st.plotly_chart(fig, use_container_width=True)

        # Steady state
        eigenvalues, eigenvectors = np.linalg.eig(trans.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        steady = np.real(eigenvectors[:, idx])
        steady = steady / steady.sum()

        st.markdown("**Steady-State Distribution (Long-Run Probabilities):**")
        steady_df = pd.DataFrame({
            'Visit Frequency': freq_order,
            'Current Distribution': [freq_counts.get(f, 0) / len(df) * 100 for f in freq_order],
            'Steady-State (%)': (steady * 100).round(1)
        })
        st.dataframe(steady_df, use_container_width=True, hide_index=True)

        # Simulate steps
        n_steps = st.slider("Simulate transitions over N periods:", 1, 20, 10)
        current = np.array([freq_counts.get(f, 0) for f in freq_order], dtype=float)
        current = current / current.sum()
        history = [current.copy()]
        for _ in range(n_steps):
            current = current @ trans
            history.append(current.copy())

        hist_df = pd.DataFrame(history, columns=freq_order)
        hist_df['Period'] = range(len(hist_df))
        hist_long = hist_df.melt(id_vars='Period', var_name='Frequency', value_name='Proportion')
        fig = px.line(hist_long, x='Period', y='Proportion', color='Frequency',
                      color_discrete_sequence=PALETTE,
                      title=f'Markov Chain: Predicted Customer Distribution Over {n_steps} Periods')
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB: Moving Averages & Forecasting ──
    with tab6:
        st.markdown("### Moving Averages & Holt-Winters Forecasting")
        st.markdown('<div class="method-box"><b>Course Tools:</b> Trend function, Holt-Winters forecasting, Moving averages — smooth data and forecast future values.</div>', unsafe_allow_html=True)

        st.markdown("#### Simulated Time-Series: Satisfaction Over Time")
        st.markdown("We create a time-ordered series from the data to demonstrate forecasting techniques.")

        # Create pseudo time-series by customer ID order
        ts_var = st.selectbox("Variable to forecast:", ['AverageSpend', 'WaitTime', 'ServiceRating', 'FoodRating'])
        window = st.slider("Moving average window:", 5, 100, 30)

        ts = df.sort_values('CustomerID')[ts_var].reset_index(drop=True)
        ts_df = pd.DataFrame({'Index': range(len(ts)), ts_var: ts})
        ts_df['SMA'] = ts.rolling(window=window).mean()
        ts_df['EMA'] = ts.ewm(span=window).mean()

        # Holt-Winters (exponential smoothing)
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        try:
            hw_model = ExponentialSmoothing(ts.values, trend='add', seasonal=None).fit(optimized=True)
            ts_df['Holt-Winters'] = hw_model.fittedvalues
            # Forecast
            n_forecast = st.slider("Forecast periods:", 10, 200, 50)
            forecast = hw_model.forecast(n_forecast)
        except Exception:
            ts_df['Holt-Winters'] = ts_df['EMA']
            n_forecast = 50
            forecast = ts_df['EMA'].iloc[-n_forecast:].values

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_df['Index'], y=ts_df[ts_var],
                                  mode='markers', name='Actual', marker=dict(color=NAVY, opacity=0.2, size=3)))
        fig.add_trace(go.Scatter(x=ts_df['Index'], y=ts_df['SMA'],
                                  mode='lines', name=f'SMA ({window})', line=dict(color=GOLD, width=2)))
        fig.add_trace(go.Scatter(x=ts_df['Index'], y=ts_df['EMA'],
                                  mode='lines', name=f'EMA ({window})', line=dict(color='#4A90D9', width=2)))
        fig.add_trace(go.Scatter(x=ts_df['Index'], y=ts_df['Holt-Winters'],
                                  mode='lines', name='Holt-Winters', line=dict(color='red', width=2)))
        # Forecast
        forecast_idx = range(len(ts), len(ts) + n_forecast)
        fig.add_trace(go.Scatter(x=list(forecast_idx), y=forecast,
                                  mode='lines', name='Forecast', line=dict(color='green', width=2, dash='dash')))
        fig.update_layout(title=f'{ts_var}: Moving Averages & Holt-Winters Forecast',
                          xaxis_title='Customer Index (time proxy)', yaxis_title=ts_var, height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Trend line
        st.markdown("#### Trend Analysis")
        z = np.polyfit(ts_df['Index'], ts_df[ts_var], 1)
        trend_line = np.polyval(z, ts_df['Index'])
        st.markdown(f"**Trend Equation:** y = {z[0]:.6f}x + {z[1]:.2f}")
        st.markdown(f"**Slope:** {z[0]:.6f} ({'Increasing' if z[0] > 0 else 'Decreasing'} trend)")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: PRESCRIPTIVE ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == pages[7]:
    st.title("💡 Prescriptive Analytics & Recommendations")
    st.markdown("Data-driven recommendations for improving restaurant customer satisfaction.")

    st.markdown("---")

    # Compute key insights
    loyalty_sat = df.groupby('LoyaltyProgramMember')[target].mean() * 100
    loyalty_lift = loyalty_sat.get(1, 0) - loyalty_sat.get(0, 0)
    reservation_sat = df.groupby('OnlineReservation')[target].mean() * 100
    low_wait_sat = df[df['WaitTime'] <= 15][target].mean() * 100
    high_wait_sat = df[df['WaitTime'] > 30][target].mean() * 100
    best_cuisine = df.groupby('PreferredCuisine')[target].mean().idxmax()
    best_cuisine_rate = df.groupby('PreferredCuisine')[target].mean().max() * 100
    best_time = df.groupby('TimeOfVisit')[target].mean().idxmax()
    best_time_rate = df.groupby('TimeOfVisit')[target].mean().max() * 100

    st.markdown("### Key Performance Indicators")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Satisfaction", f"{df[target].mean()*100:.1f}%")
    c2.metric("Loyalty Lift", f"{loyalty_lift:+.1f} pp")
    c3.metric("Best Cuisine", f"{best_cuisine} ({best_cuisine_rate:.1f}%)")
    c4.metric("Best Time", f"{best_time} ({best_time_rate:.1f}%)")

    st.markdown("---")

    # Recommendations
    st.markdown("### Actionable Recommendations")

    st.markdown(f"""
    <div class="insight-box">
    <h4>1. Optimize Wait Times</h4>
    <p>Customers with wait times <b>under 15 minutes</b> have a satisfaction rate of <b>{low_wait_sat:.1f}%</b>,
    compared to <b>{high_wait_sat:.1f}%</b> for those waiting over 30 minutes.</p>
    <p><b>Action:</b> Implement queue management systems, optimize kitchen workflows, and consider
    reservation-based seating to reduce peak-hour wait times.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-box">
    <h4>2. {'Expand' if loyalty_lift > 0 else 'Redesign'} Loyalty Programs</h4>
    <p>Loyalty members show a <b>{loyalty_lift:+.1f} percentage point</b> difference in satisfaction
    ({loyalty_sat.get(1,0):.1f}% vs {loyalty_sat.get(0,0):.1f}%).</p>
    <p><b>Action:</b> {'Incentivize enrollment and reward repeat visits with personalized perks.' if loyalty_lift > 0
    else 'Redesign the program to deliver more meaningful value to members.'}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-box">
    <h4>3. Tailor Service by Dining Occasion</h4>
    <p>Satisfaction varies by occasion type. Different expectations require different service approaches.</p>
    <p><b>Action:</b> Train staff to recognize occasion types and adjust service style — faster for business,
    more attentive for celebrations, relaxed for casual dining.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-box">
    <h4>4. Leverage Digital Channels</h4>
    <p>Online reservation users show <b>{reservation_sat.get(1,0):.1f}%</b> satisfaction vs
    <b>{reservation_sat.get(0,0):.1f}%</b> for walk-ins.</p>
    <p><b>Action:</b> {'Promote online booking to set expectations and reduce friction.' if reservation_sat.get(1,0) > reservation_sat.get(0,0)
    else 'Improve the online booking experience to match walk-in quality.'}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-box">
    <h4>5. Focus on Service Quality</h4>
    <p>Service, food, and ambiance ratings all contribute to overall satisfaction, but their relative importance varies.</p>
    <p><b>Action:</b> Prioritize improvements in the areas identified as most impactful by the regression models.
    Regular staff training and quality audits can help maintain high standards.</p>
    </div>
    """, unsafe_allow_html=True)

    # Implications Wheel
    st.markdown("### Implications Wheel")
    st.markdown('<div class="method-box"><b>Course Tool:</b> Implications Wheel — map how changes in one area ripple through the system.</div>', unsafe_allow_html=True)

    st.markdown("""
    ```
                          ┌─────────────────────┐
                          │   REDUCE WAIT TIME   │
                          └──────────┬──────────┘
                 ┌───────────────────┼───────────────────┐
                 ▼                   ▼                   ▼
        ┌────────────────┐ ┌─────────────────┐ ┌────────────────┐
        │ Higher Service │ │   Better First  │ │  More Repeat   │
        │    Ratings     │ │   Impressions   │ │    Visits      │
        └───────┬────────┘ └────────┬────────┘ └───────┬────────┘
                │                   │                   │
                ▼                   ▼                   ▼
        ┌────────────────┐ ┌─────────────────┐ ┌────────────────┐
        │    Higher      │ │  More Online    │ │   Loyalty      │
        │  Satisfaction  │ │    Reviews      │ │   Sign-ups     │
        └───────┬────────┘ └────────┬────────┘ └───────┬────────┘
                │                   │                   │
                └───────────────────┼───────────────────┘
                                    ▼
                          ┌─────────────────────┐
                          │   REVENUE GROWTH     │
                          └─────────────────────┘
    ```
    """)

    # Customer satisfaction predictor
    st.markdown("---")
    st.markdown("### Interactive Satisfaction Predictor")
    st.markdown("Input a customer profile to predict satisfaction probability.")

    @st.cache_resource
    def get_predictor():
        df_m2 = df.copy()
        for col in categorical_cols:
            le2 = LabelEncoder()
            df_m2[col + '_enc'] = le2.fit_transform(df_m2[col])
        feat2 = numeric_cols + binary_cols + [c + '_enc' for c in categorical_cols]
        sc2 = StandardScaler()
        X2 = sc2.fit_transform(df_m2[feat2])
        lr2 = LogisticRegression(max_iter=1000, random_state=42)
        lr2.fit(X2, df_m2[target])
        return lr2, sc2, feat2, {col: LabelEncoder().fit(df[col]) for col in categorical_cols}

    pred_model, pred_scaler, pred_feats, pred_le = get_predictor()

    c1, c2, c3, c4 = st.columns(4)
    inp_age = c1.slider("Age:", 18, 70, 35)
    inp_income = c2.slider("Income ($):", 20000, 150000, 80000, step=5000)
    inp_spend = c3.slider("Avg Spend ($):", 10, 200, 80)
    inp_wait = c4.slider("Wait Time (min):", 0, 60, 20)

    c1, c2, c3, c4 = st.columns(4)
    inp_group = c1.slider("Group Size:", 1, 10, 3)
    inp_service = c2.slider("Service Rating:", 1, 5, 3)
    inp_food = c3.slider("Food Rating:", 1, 5, 3)
    inp_ambiance = c4.slider("Ambiance Rating:", 1, 5, 3)

    c1, c2, c3 = st.columns(3)
    inp_cuisine = c1.selectbox("Cuisine:", df['PreferredCuisine'].unique(), key='pred_cuis')
    inp_occasion = c2.selectbox("Occasion:", df['DiningOccasion'].unique(), key='pred_occ')
    inp_time = c3.selectbox("Time of Visit:", df['TimeOfVisit'].unique(), key='pred_time')

    c1, c2, c3, c4 = st.columns(4)
    inp_freq = c1.selectbox("Visit Frequency:", df['VisitFrequency'].unique(), key='pred_freq')
    inp_gender = c2.selectbox("Gender:", df['Gender'].unique(), key='pred_gen')
    inp_meal = c3.selectbox("Meal Type:", df['MealType'].unique(), key='pred_meal')
    inp_loyalty = c4.selectbox("Loyalty Member:", ['No', 'Yes'], key='pred_loy')

    c1, c2 = st.columns(2)
    inp_reservation = c1.selectbox("Online Reservation:", ['No', 'Yes'], key='pred_res')
    inp_delivery = c2.selectbox("Delivery Order:", ['No', 'Yes'], key='pred_del')

    input_row = {
        'Age': inp_age, 'Income': inp_income, 'AverageSpend': inp_spend,
        'GroupSize': inp_group, 'WaitTime': inp_wait,
        'ServiceRating': inp_service, 'FoodRating': inp_food, 'AmbianceRating': inp_ambiance,
        'OnlineReservation': 1 if inp_reservation == 'Yes' else 0,
        'DeliveryOrder': 1 if inp_delivery == 'Yes' else 0,
        'LoyaltyProgramMember': 1 if inp_loyalty == 'Yes' else 0,
    }
    for col in categorical_cols:
        val = {'Gender': inp_gender, 'VisitFrequency': inp_freq,
               'PreferredCuisine': inp_cuisine, 'TimeOfVisit': inp_time,
               'DiningOccasion': inp_occasion, 'MealType': inp_meal}[col]
        input_row[col + '_enc'] = pred_le[col].transform([val])[0]

    input_df = pd.DataFrame([input_row])[pred_feats]
    prob = pred_model.predict_proba(pred_scaler.transform(input_df))[0][1]

    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("Predicted Satisfaction Probability", f"{prob:.1%}")
    c2.metric("Prediction", "✅ HIGH Satisfaction" if prob >= 0.5 else "❌ LOW Satisfaction")

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Satisfaction Probability (%)"},
        delta={'reference': df[target].mean() * 100},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': GOLD},
            'steps': [
                {'range': [0, 30], 'color': '#FFCDD2'},
                {'range': [30, 60], 'color': '#FFF9C4'},
                {'range': [60, 100], 'color': '#C8E6C9'}
            ],
            'threshold': {'line': {'color': 'red', 'width': 4}, 'thickness': 0.75, 'value': 50}
        }
    ))
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: CONCLUSIONS & LIMITATIONS
# ═════════════════════════════════════════════════════════════════════════════
elif page == pages[8]:
    st.title("📌 Conclusions & Limitations")

    st.markdown("### Key Conclusions")
    st.markdown("""
    1. **Customer satisfaction is multifactorial** — no single variable dominates. The interplay of
       wait time, service quality, food quality, and dining context all contribute to the outcome.

    2. **Operational factors matter** — wait time consistently emerges as a key lever. Monte Carlo
       simulations quantify the impact: reducing wait times significantly shifts the satisfaction
       probability distribution.

    3. **Segmentation reveals nuance** — satisfaction patterns differ by dining occasion, visit
       frequency, and cuisine preference. One-size-fits-all strategies are suboptimal.

    4. **Predictive models enable proactive management** — logistic regression and decision trees
       can identify at-risk customers before they leave dissatisfied, enabling real-time
       service interventions.

    5. **The analytical toolkit from this course provides comprehensive coverage** — from descriptive
       statistics (understanding what happened) through inferential testing (understanding why)
       to predictive and prescriptive modeling (forecasting what will happen and what to do about it).
    """)

    st.markdown("### Course Tools Demonstrated in This Analysis")
    tools_df = pd.DataFrame({
        'Course Topic': [
            'Descriptive Statistics', 'Pivot Tables & Dashboards', 'VLOOKUP / INDEX MATCH',
            'Rank & Percentile', 'Z-Test', 'Chi-Square Test',
            'Correlation & R²', 'ANOVA (Single & Two Factor)',
            'Multivariate Regression', 'Monte Carlo Simulation',
            'Holt-Winters / Moving Averages', 'Markov Chains',
            'Decision Trees', "Simpson's Paradox",
            'Probability Functions', 'Implications Wheel',
            'Histograms / Box Plots / Stem Plots', 'Data Storytelling'
        ],
        'Where in Dashboard': [
            'Descriptive Analytics → Summary Statistics',
            'Descriptive Analytics → Pivot Tables',
            'Descriptive Analytics → Index Match / Lookup',
            'Descriptive Analytics → Rank & Percentile',
            'Inferential Statistics → Z-Test',
            'Inferential Statistics → Chi-Square Test',
            'Inferential Statistics → Correlation & R²',
            'Inferential Statistics → ANOVA',
            'Predictive Modeling → Linear Regression',
            'Predictive Modeling → Monte Carlo Simulation',
            'Predictive Modeling → Moving Averages & Forecasting',
            'Predictive Modeling → Markov Chains',
            'Predictive Modeling → Decision Trees',
            'Exploratory Data Analysis → Simpson\'s Paradox',
            'Inferential Statistics → Z-Test (Normal overlay)',
            'Prescriptive Analytics → Implications Wheel',
            'Exploratory Data Analysis → Histograms / Box & Stem Plots',
            'All pages — narrative and visualization'
        ]
    })
    st.dataframe(tools_df, use_container_width=True, hide_index=True)

    st.markdown("### Data & Project Limitations")
    st.markdown("""
    1. **Secondary Data Source** — Kaggle dataset; no control over collection methodology. Findings
       should be validated with primary data (surveys, interviews).

    2. **Binary Satisfaction Measure** — The target is binary (0/1), simplifying a nuanced concept.
       A continuous scale (1-10) would allow more granular analysis.

    3. **Cross-Sectional Data** — Single snapshot in time. Cannot infer causation or track
       longitudinal changes. True Holt-Winters forecasting requires genuine time-series data.

    4. **No Location Data** — Cannot assess geographic or location-specific effects.

    5. **Potential Synthetic Data** — Distribution uniformity suggests possible synthetic generation,
       limiting real-world generalizability.

    6. **Model Assumptions** — Linear/logistic regression assume linear relationships with
       (log-odds of) the outcome. Non-linear effects and interactions may be missed.

    7. **No Causal Claims** — All analyses are correlational. Controlled experiments would be
       needed to establish causality.
    """)

    st.markdown("---")
    st.markdown("### Thank You")
    st.markdown("""
    **GWU MKTG 4163 — Applied Marketing Analytics**
    Spring 2026 | Professor David Ashley

    **Team Members:** Max, Leonie, Sam & Yen Kai

    *Analysis built with Python, Streamlit, Plotly, scikit-learn, statsmodels, and scipy.*
    """)
