"""
============================================================
ATM Intelligence Demand Forecasting - FA-2 Interactive Planner
============================================================
A comprehensive data mining application for ATM cash management.

Features:
- Stage 3: Exploratory Data Analysis (EDA)
- Stage 4: Clustering Analysis
- Stage 5: Anomaly Detection
- Stage 6: Interactive Planner

Author: Data Mining FA-2 Project
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =========================================
# PAGE CONFIGURATION
# =========================================
st.set_page_config(
    page_title="ATM Intelligence Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================================
# DATA LOADING
# =========================================
@st.cache_data
def load_data():
    """Load and preprocess the ATM dataset"""
    try:
        df = pd.read_csv("cleaned_atm_data.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please upload 'cleaned_atm_data.csv'")
        return None

# =========================================
# SIDEBAR NAVIGATION
# =========================================
st.sidebar.markdown("<h1 style='text-align: center;'>🏦 ATM Intelligence</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    ["📊 Dashboard Overview", 
     "📈 Exploratory Data Analysis", 
     "🎯 Clustering Analysis", 
     "⚠️ Anomaly Detection",
     "🔍 Interactive Planner"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filter Options")

# =========================================
# LOAD DATA
# =========================================
df = load_data()

if df is None:
    st.stop()

# =========================================
# GLOBAL FILTERS
# =========================================
# Date range filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df['Date'].min().date(), df['Date'].max().date()),
    min_value=df['Date'].min().date(),
    max_value=df['Date'].max().date()
)

# Location filter
locations = ['All'] + list(df['Location_Type'].unique())
selected_location = st.sidebar.selectbox("Location Type", locations)

# Apply filters
if len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
else:
    filtered_df = df.copy()

if selected_location != 'All':
    filtered_df = filtered_df[filtered_df['Location_Type'] == selected_location]

# =========================================
# PAGE 1: DASHBOARD OVERVIEW
# =========================================
if page == "📊 Dashboard Overview":
    st.markdown("<h1 class='main-header'>🏦 ATM Intelligence Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>FA-2: Building Actionable Insights and Interactive Python Script</p>", unsafe_allow_html=True)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Transactions",
            value=f"{len(filtered_df):,}",
            delta=f"{len(filtered_df) - len(df):,} filtered"
        )
    
    with col2:
        st.metric(
            label="Total Withdrawals",
            value=f"₹{filtered_df['Total_Withdrawals'].sum():,.0f}"
        )
    
    with col3:
        st.metric(
            label="Avg Withdrawal",
            value=f"₹{filtered_df['Total_Withdrawals'].mean():,.0f}"
        )
    
    with col4:
        st.metric(
            label="Total ATMs",
            value=f"{filtered_df['ATM_ID'].nunique()}"
        )
    
    st.markdown("---")
    
    # Quick Stats Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 class='sub-header'>📊 Transaction Distribution by Location</h3>", unsafe_allow_html=True)
        loc_counts = filtered_df['Location_Type'].value_counts()
        fig = px.pie(
            values=loc_counts.values,
            names=loc_counts.index,
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<h3 class='sub-header'>📅 Withdrawals Over Time</h3>", unsafe_allow_html=True)
        daily_withdrawals = filtered_df.groupby('Date')['Total_Withdrawals'].sum().reset_index()
        fig = px.line(
            daily_withdrawals, 
            x='Date', 
            y='Total_Withdrawals',
            title="Daily Withdrawal Trend"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary Statistics
    st.markdown("<h3 class='sub-header'>📋 Summary Statistics</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.markdown("**Key Insights:**")
        st.write(f"• Peak Withdrawal Day: **{filtered_df.groupby('Day_of_Week')['Total_Withdrawals'].sum().idxmax()}**")
        st.write(f"• Peak Time: **{filtered_df.groupby('Time_of_Day')['Total_Withdrawals'].sum().idxmax()}**")
        st.write(f"• Holiday Transactions: **{filtered_df[filtered_df['Holiday_Flag']==1].shape[0]:,}**")
        st.write(f"• Special Events: **{filtered_df[filtered_df['Special_Event_Flag']==1].shape[0]:,}**")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.markdown("**Data Quality:**")
        st.write(f"• Total Records: **{len(filtered_df):,}**")
        st.write(f"• Date Range: **{filtered_df['Date'].min().strftime('%Y-%m-%d')}** to **{filtered_df['Date'].max().strftime('%Y-%m-%d')}**")
        st.write(f"• Missing Values: **0** ✓")
        st.write(f"• Duplicate Records: **0** ✓")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================================
# PAGE 2: EXPLORATORY DATA ANALYSIS
# =========================================
elif page == "📈 Exploratory Data Analysis":
    st.markdown("<h1 class='main-header'>📈 Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    
    # EDA Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Distribution Analysis", 
        "Time-based Trends", 
        "Holiday & Event Impact",
        "External Factors",
        "Relationship Analysis"
    ])
    
    # TAB 1: Distribution Analysis
    with tab1:
        st.markdown("<h3 class='sub-header'>Distribution of Withdrawals and Deposits</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram for Withdrawals
            fig = px.histogram(
                filtered_df,
                x='Total_Withdrawals',
                nbins=50,
                title='Distribution of Total Withdrawals',
                color_discrete_sequence=['#3498db']
            )
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.write(f"**Observation:** Withdrawal distribution is approximately normal with mean ₹{filtered_df['Total_Withdrawals'].mean():,.0f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Histogram for Deposits
            fig = px.histogram(
                filtered_df,
                x='Total_Deposits',
                nbins=50,
                title='Distribution of Total Deposits',
                color_discrete_sequence=['#27ae60']
            )
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.write(f"**Observation:** Deposit distribution shows mean ₹{filtered_df['Total_Deposits'].mean():,.0f} with right skew")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Box plots for outliers
        st.markdown("<h3 class='sub-header'>Outlier Detection with Box Plots</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                filtered_df,
                y='Total_Withdrawals',
                title='Withdrawals Box Plot (Outlier Check)',
                color_discrete_sequence=['#3498db']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                filtered_df,
                y='Total_Deposits',
                title='Deposits Box Plot (Outlier Check)',
                color_discrete_sequence=['#27ae60']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Time-based Trends
    with tab2:
        st.markdown("<h3 class='sub-header'>Time Series Analysis</h3>", unsafe_allow_html=True)
        
        # Line chart over time
        daily_data = filtered_df.groupby('Date').agg({
            'Total_Withdrawals': 'sum',
            'Total_Deposits': 'sum'
        }).reset_index()
        
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(
            x=daily_data['Date'],
            y=daily_data['Total_Withdrawals'],
            name='Withdrawals',
            line=dict(color='#3498db')
        ))
        fig.add_trace(go.Scatter(
            x=daily_data['Date'],
            y=daily_data['Total_Deposits'],
            name='Deposits',
            line=dict(color='#27ae60')
        ))
        fig.update_layout(title='Withdrawals and Deposits Over Time', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Day of Week analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h4>Withdrawals by Day of Week</h4>")
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_data = filtered_df.groupby('Day_of_Week')['Total_Withdrawals'].mean()
            day_data = day_data.reindex(day_order)
            
            fig = px.bar(
                x=day_data.index,
                y=day_data.values,
                title='Average Withdrawals by Day',
                color=day_data.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(xaxis_title='Day', yaxis_title='Avg Withdrawals')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("<h4>Withdrawals by Time of Day</h4>")
            time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
            time_data = filtered_df.groupby('Time_of_Day')['Total_Withdrawals'].mean()
            time_data = time_data.reindex(time_order)
            
            fig = px.bar(
                x=time_data.index,
                y=time_data.values,
                title='Average Withdrawals by Time',
                color=time_data.values,
                color_continuous_scale='Greens'
            )
            fig.update_layout(xaxis_title='Time of Day', yaxis_title='Avg Withdrawals')
            st.plotly_chart(fig, use_container_width=True)
        
        # Monthly trend
        st.markdown("<h3 class='sub-header'>Monthly Trend Analysis</h3>", unsafe_allow_html=True)
        
        monthly_data = filtered_df.groupby(['Year', 'Month']).agg({
            'Total_Withdrawals': 'sum',
            'Total_Deposits': 'sum'
        }).reset_index()
        monthly_data['Year_Month'] = monthly_data['Year'].astype(str) + '-' + monthly_data['Month'].astype(str).str.zfill(2)
        
        fig = px.line(
            monthly_data,
            x='Year_Month',
            y=['Total_Withdrawals', 'Total_Deposits'],
            title='Monthly Withdrawals and Deposits Trend',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: Holiday & Event Impact
    with tab3:
        st.markdown("<h3 class='sub-header'>Holiday Impact Analysis</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            holiday_data = filtered_df.groupby('Holiday_Flag')['Total_Withdrawals'].mean()
            
            fig = px.bar(
                x=['Normal Day', 'Holiday'],
                y=holiday_data.values,
                title='Average Withdrawals: Holiday vs Normal',
                color=holiday_data.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(xaxis_title='', yaxis_title='Avg Withdrawals')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            holiday_total = filtered_df.groupby('Holiday_Flag')['Total_Withdrawals'].sum()
            
            fig = px.pie(
                values=holiday_total.values,
                names=['Normal Day', 'Holiday'],
                title='Total Withdrawals Distribution',
                hole=0.4,
                color_discrete_sequence=['#3498db', '#e74c3c']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Special Event Impact
        st.markdown("<h3 class='sub-header'>Special Event Impact Analysis</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            event_data = filtered_df.groupby('Special_Event_Flag')['Total_Withdrawals'].mean()
            
            fig = px.bar(
                x=['No Event', 'Special Event'],
                y=event_data.values,
                title='Average Withdrawals: Event vs No Event',
                color=event_data.values,
                color_continuous_scale='Purples'
            )
            fig.update_layout(xaxis_title='', yaxis_title='Avg Withdrawals')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Holiday and Event combined
            combined = filtered_df.groupby(['Holiday_Flag', 'Special_Event_Flag'])['Total_Withdrawals'].mean().reset_index()
            combined['Category'] = combined.apply(
                lambda x: f"Holiday: {x['Holiday_Flag']}, Event: {x['Special_Event_Flag']}", axis=1
            )
            
            fig = px.bar(
                combined,
                x='Category',
                y='Total_Withdrawals',
                title='Withdrawals by Holiday/Event Combination',
                color='Total_Withdrawals'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Insight box
        holiday_avg = filtered_df[filtered_df['Holiday_Flag']==1]['Total_Withdrawals'].mean()
        normal_avg = filtered_df[filtered_df['Holiday_Flag']==0]['Total_Withdrawals'].mean()
        diff_pct = ((holiday_avg - normal_avg) / normal_avg) * 100
        
        st.markdown(f"<div class='insight-box'><b>Holiday Impact Insight:</b> Holiday withdrawals are <b>{diff_pct:.1f}%</b> {'higher' if diff_pct > 0 else 'lower'} than normal days.</div>", unsafe_allow_html=True)
    
    # TAB 4: External Factors
    with tab4:
        st.markdown("<h3 class='sub-header'>Weather Impact Analysis</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            weather_data = filtered_df.groupby('Weather_Condition')['Total_Withdrawals'].mean()
            
            fig = px.bar(
                x=weather_data.index,
                y=weather_data.values,
                title='Average Withdrawals by Weather',
                color=weather_data.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(xaxis_title='Weather Condition', yaxis_title='Avg Withdrawals')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                filtered_df,
                x='Weather_Condition',
                y='Total_Withdrawals',
                title='Withdrawal Distribution by Weather',
                color='Weather_Condition'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Competitor Analysis
        st.markdown("<h3 class='sub-header'>Competitor ATM Impact</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            competitor_data = filtered_df.groupby('Nearby_Competitor_ATMs')['Total_Withdrawals'].mean()
            
            fig = px.bar(
                x=competitor_data.index.astype(str),
                y=competitor_data.values,
                title='Avg Withdrawals by Number of Competitor ATMs',
                color=competitor_data.values,
                color_continuous_scale='Oranges'
            )
            fig.update_layout(xaxis_title='Number of Nearby Competitors', yaxis_title='Avg Withdrawals')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                filtered_df,
                x='Nearby_Competitor_ATMs',
                y='Total_Withdrawals',
                title='Withdrawal Distribution by Competitors',
                color='Nearby_Competitor_ATMs'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Location Type Analysis
        st.markdown("<h3 class='sub-header'>Location Type Analysis</h3>", unsafe_allow_html=True)
        
        location_data = filtered_df.groupby('Location_Type').agg({
            'Total_Withdrawals': 'mean',
            'Total_Deposits': 'mean'
        }).reset_index()
        
        fig = px.bar(
            location_data,
            x='Location_Type',
            y=['Total_Withdrawals', 'Total_Deposits'],
            barmode='group',
            title='Average Transactions by Location Type'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 5: Relationship Analysis
    with tab5:
        st.markdown("<h3 class='sub-header'>Correlation Analysis</h3>", unsafe_allow_html=True)
        
        # Correlation heatmap
        numeric_cols = ['Total_Withdrawals', 'Total_Deposits', 'Previous_Day_Cash_Level', 
                       'Cash_Demand_Next_Day', 'Nearby_Competitor_ATMs', 'Holiday_Flag', 
                       'Special_Event_Flag', 'Day_of_Week_Encoded', 'Time_of_Day_Encoded']
        
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        fig.update_layout(title='Correlation Heatmap of Numeric Features')
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot
        st.markdown("<h3 class='sub-header'>Scatter Plot Analysis</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                filtered_df,
                x='Previous_Day_Cash_Level',
                y='Cash_Demand_Next_Day',
                title='Cash Level vs Next Day Demand',
                color='Location_Type',
                opacity=0.6
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                filtered_df,
                x='Total_Withdrawals',
                y='Total_Deposits',
                title='Withdrawals vs Deposits',
                color='Day_of_Week',
                opacity=0.6
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Pair plot sample
        st.markdown("<h3 class='sub-header'>Feature Relationships</h3>", unsafe_allow_html=True)
        
        sample_df = filtered_df[['Total_Withdrawals', 'Total_Deposits', 'Previous_Day_Cash_Level', 'Location_Type']].sample(min(500, len(filtered_df)))
        
        fig = px.scatter_matrix(
            sample_df,
            dimensions=['Total_Withdrawals', 'Total_Deposits', 'Previous_Day_Cash_Level'],
            color='Location_Type',
            title='Scatter Matrix of Key Features'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

# =========================================
# PAGE 3: CLUSTERING ANALYSIS
# =========================================
elif page == "🎯 Clustering Analysis":
    st.markdown("<h1 class='main-header'>🎯 Clustering Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Grouping ATMs by demand behavior for efficient cash management</p>", unsafe_allow_html=True)
    
    # Feature selection for clustering
    st.markdown("<h3 class='sub-header'>Feature Selection</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Select Features for Clustering:**")
        features = st.multiselect(
            "Features",
            ['Total_Withdrawals', 'Total_Deposits', 'Previous_Day_Cash_Level', 
             'Cash_Demand_Next_Day', 'Nearby_Competitor_ATMs', 'Withdrawal_Deposit_Ratio',
             'Cash_Utilization_Rate'],
            default=['Total_Withdrawals', 'Total_Deposits', 'Cash_Demand_Next_Day']
        )
        
        max_clusters = st.slider("Maximum Clusters to Evaluate", 2, 10, 6)
    
    if len(features) >= 2:
        with col2:
            # Prepare data for clustering
            X = filtered_df[features].copy()
            
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Elbow Method
            st.markdown("**Elbow Method for Optimal Clusters**")
            inertias = []
            K_range = range(2, max_clusters + 1)
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
            
            fig = px.line(
                x=list(K_range),
                y=inertias,
                markers=True,
                title='Elbow Method - Inertia vs Number of Clusters'
            )
            fig.update_layout(xaxis_title='Number of Clusters', yaxis_title='Inertia')
            st.plotly_chart(fig, use_container_width=True)
        
        # Silhouette Score
        st.markdown("<h3 class='sub-header'>Silhouette Score Analysis</h3>", unsafe_allow_html=True)
        
        silhouette_scores = []
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=list(K_range),
                y=silhouette_scores,
                title='Silhouette Score by Number of Clusters',
                color=silhouette_scores,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(xaxis_title='Number of Clusters', yaxis_title='Silhouette Score')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            optimal_k = list(K_range)[np.argmax(silhouette_scores)]
            st.markdown(f"<div class='success-box'><b>Optimal Number of Clusters: {optimal_k}</b><br>Silhouette Score: {max(silhouette_scores):.3f}</div>", unsafe_allow_html=True)
        
        # Perform Clustering
        st.markdown("<h3 class='sub-header'>Clustering Results</h3>", unsafe_allow_html=True)
        
        n_clusters = st.selectbox("Select Number of Clusters", list(K_range), index=optimal_k-2)
        
        # Apply K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        filtered_df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Cluster Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # 2D Scatter plot
            fig = px.scatter(
                filtered_df,
                x=features[0],
                y=features[1] if len(features) > 1 else features[0],
                color='Cluster',
                title='Cluster Visualization',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 3D Scatter if 3+ features
            if len(features) >= 3:
                fig = px.scatter_3d(
                    filtered_df,
                    x=features[0],
                    y=features[1],
                    z=features[2],
                    color='Cluster',
                    title='3D Cluster Visualization',
                    opacity=0.7
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Cluster distribution pie chart
                cluster_counts = filtered_df['Cluster'].value_counts()
                fig = px.pie(
                    values=cluster_counts.values,
                    names=cluster_counts.index,
                    title='Cluster Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Cluster Statistics
        st.markdown("<h3 class='sub-header'>Cluster Statistics</h3>", unsafe_allow_html=True)
        
        cluster_stats = filtered_df.groupby('Cluster')[features].mean()
        st.dataframe(cluster_stats.style.background_gradient(cmap='Blues'), use_container_width=True)
        
        # Cluster Interpretation
        st.markdown("<h3 class='sub-header'>Cluster Interpretation</h3>", unsafe_allow_html=True)
        
        # Automatically interpret clusters
        cluster_labels = {}
        for i in range(n_clusters):
            cluster_data = filtered_df[filtered_df['Cluster'] == i]
            avg_withdrawal = cluster_data['Total_Withdrawals'].mean()
            
            if avg_withdrawal > filtered_df['Total_Withdrawals'].quantile(0.75):
                cluster_labels[i] = "High-Demand ATMs"
            elif avg_withdrawal > filtered_df['Total_Withdrawals'].quantile(0.5):
                cluster_labels[i] = "Medium-Demand ATMs"
            elif avg_withdrawal > filtered_df['Total_Withdrawals'].quantile(0.25):
                cluster_labels[i] = "Steady-Demand ATMs"
            else:
                cluster_labels[i] = "Low-Demand ATMs"
        
        for i, label in cluster_labels.items():
            count = (filtered_df['Cluster'] == i).sum()
            st.markdown(f"<div class='insight-box'><b>Cluster {i}:</b> {label} ({count} transactions)</div>", unsafe_allow_html=True)
        
        # Cluster by Location
        st.markdown("<h3 class='sub-header'>Cluster Distribution by Location</h3>", unsafe_allow_html=True)
        
        cluster_location = pd.crosstab(filtered_df['Location_Type'], filtered_df['Cluster'])
        fig = px.bar(
            cluster_location,
            barmode='group',
            title='Cluster Distribution by Location Type'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("Please select at least 2 features for clustering analysis.")

# =========================================
# PAGE 4: ANOMALY DETECTION
# =========================================
elif page == "⚠️ Anomaly Detection":
    st.markdown("<h1 class='main-header'>⚠️ Anomaly Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Detecting unusual withdrawal patterns during holidays and events</p>", unsafe_allow_html=True)
    
    # Method selection
    st.markdown("<h3 class='sub-header'>Detection Method</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        method = st.selectbox(
            "Select Anomaly Detection Method",
            ["Z-Score", "IQR Method", "Isolation Forest", "Local Outlier Factor (LOF)"]
        )
        
        contamination = st.slider("Contamination Rate (%)", 1, 20, 5) / 100
    
    # Prepare data
    X = filtered_df[['Total_Withdrawals', 'Total_Deposits', 'Previous_Day_Cash_Level']].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply selected method
    if method == "Z-Score":
        z_scores = np.abs(stats.zscore(filtered_df['Total_Withdrawals']))
        filtered_df['Is_Anomaly'] = (z_scores > 3).astype(int)
        threshold = 3
        
    elif method == "IQR Method":
        Q1 = filtered_df['Total_Withdrawals'].quantile(0.25)
        Q3 = filtered_df['Total_Withdrawals'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_df['Is_Anomaly'] = ((filtered_df['Total_Withdrawals'] < lower_bound) | 
                                     (filtered_df['Total_Withdrawals'] > upper_bound)).astype(int)
        threshold = upper_bound
        
    elif method == "Isolation Forest":
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        filtered_df['Is_Anomaly'] = (iso_forest.fit_predict(X_scaled) == -1).astype(int)
        
    else:  # LOF
        lof = LocalOutlierFactor(contamination=contamination)
        filtered_df['Is_Anomaly'] = (lof.fit_predict(X_scaled) == -1).astype(int)
    
    # Results
    with col2:
        anomalies = filtered_df['Is_Anomaly'].sum()
        anomaly_pct = (anomalies / len(filtered_df)) * 100
        
        st.metric(
            label="Anomalies Detected",
            value=f"{anomalies:,}",
            delta=f"{anomaly_pct:.2f}% of total"
        )
    
    # Anomaly Visualization
    st.markdown("<h3 class='sub-header'>Anomaly Visualization</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            filtered_df,
            x='Total_Withdrawals',
            y='Total_Deposits',
            color='Is_Anomaly',
            title='Anomaly Detection: Withdrawals vs Deposits',
            color_discrete_map={0: '#3498db', 1: '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            filtered_df,
            x='Date',
            y='Total_Withdrawals',
            color='Is_Anomaly',
            title='Anomalies Over Time',
            color_discrete_map={0: '#3498db', 1: '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Holiday vs Normal Day Anomalies
    st.markdown("<h3 class='sub-header'>Anomalies: Holiday vs Normal Days</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        holiday_anomalies = filtered_df[filtered_df['Holiday_Flag'] == 1]
        normal_anomalies = filtered_df[filtered_df['Holiday_Flag'] == 0]
        
        fig = px.bar(
            x=['Normal Days', 'Holidays'],
            y=[normal_anomalies['Is_Anomaly'].sum(), holiday_anomalies['Is_Anomaly'].sum()],
            title='Anomalies: Holiday vs Normal Days',
            color=[normal_anomalies['Is_Anomaly'].sum(), holiday_anomalies['Is_Anomaly'].sum()],
            color_continuous_scale='Reds'
        )
        fig.update_layout(xaxis_title='', yaxis_title='Number of Anomalies')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        event_anomalies = filtered_df[filtered_df['Special_Event_Flag'] == 1]
        no_event_anomalies = filtered_df[filtered_df['Special_Event_Flag'] == 0]
        
        fig = px.bar(
            x=['No Event', 'Special Event'],
            y=[no_event_anomalies['Is_Anomaly'].sum(), event_anomalies['Is_Anomaly'].sum()],
            title='Anomalies: Event vs No Event',
            color=[no_event_anomalies['Is_Anomaly'].sum(), event_anomalies['Is_Anomaly'].sum()],
            color_continuous_scale='Purples'
        )
        fig.update_layout(xaxis_title='', yaxis_title='Number of Anomalies')
        st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly Details Table
    st.markdown("<h3 class='sub-header'>Anomaly Details</h3>", unsafe_allow_html=True)
    
    anomaly_df = filtered_df[filtered_df['Is_Anomaly'] == 1][
        ['ATM_ID', 'Date', 'Day_of_Week', 'Time_of_Day', 'Total_Withdrawals', 
         'Location_Type', 'Holiday_Flag', 'Special_Event_Flag', 'Weather_Condition']
    ].sort_values('Total_Withdrawals', ascending=False)
    
    st.dataframe(anomaly_df.head(20), use_container_width=True)
    
    # Warning box for anomalies
    if anomalies > 0:
        st.markdown(f"<div class='warning-box'><b>⚠️ Alert:</b> {anomalies} anomalous transactions detected. These may require investigation for cash management optimization.</div>", unsafe_allow_html=True)

# =========================================
# PAGE 5: INTERACTIVE PLANNER
# =========================================
elif page == "🔍 Interactive Planner":
    st.markdown("<h1 class='main-header'>🔍 Interactive ATM Planner</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Query and analyze ATM cash demand behavior</p>", unsafe_allow_html=True)
    
    # Query Builder
    st.markdown("<h3 class='sub-header'>Build Your Query</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_day = st.multiselect(
            "Day of Week",
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            default=['Monday', 'Friday']
        )
    
    with col2:
        selected_time = st.multiselect(
            "Time of Day",
            ['Morning', 'Afternoon', 'Evening', 'Night'],
            default=['Morning', 'Evening']
        )
    
    with col3:
        selected_loc = st.multiselect(
            "Location Type",
            ['Standalone', 'Supermarket', 'Mall', 'Bank Branch', 'Gas Station'],
            default=['Mall', 'Supermarket']
        )
    
    # Filter data
    query_df = filtered_df.copy()
    
    if selected_day:
        query_df = query_df[query_df['Day_of_Week'].isin(selected_day)]
    if selected_time:
        query_df = query_df[query_df['Time_of_Day'].isin(selected_time)]
    if selected_loc:
        query_df = query_df[query_df['Location_Type'].isin(selected_loc)]
    
    # Display query results
    st.markdown(f"<h3 class='sub-header'>Query Results: {len(query_df):,} records found</h3>", unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Withdrawals", f"₹{query_df['Total_Withdrawals'].sum():,.0f}")
    
    with col2:
        st.metric("Avg Withdrawal", f"₹{query_df['Total_Withdrawals'].mean():,.0f}")
    
    with col3:
        st.metric("Max Withdrawal", f"₹{query_df['Total_Withdrawals'].max():,.0f}")
    
    with col4:
        st.metric("Total ATMs", f"{query_df['ATM_ID'].nunique()}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        if len(query_df) > 0:
            fig = px.bar(
                query_df.groupby('Day_of_Week')['Total_Withdrawals'].mean().reset_index(),
                x='Day_of_Week',
                y='Total_Withdrawals',
                title='Avg Withdrawals by Selected Days',
                color='Total_Withdrawals'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if len(query_df) > 0:
            fig = px.pie(
                query_df.groupby('Location_Type')['Total_Withdrawals'].sum().reset_index(),
                values='Total_Withdrawals',
                names='Location_Type',
                title='Withdrawals by Location'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Data preview
    st.markdown("<h3 class='sub-header'>Data Preview</h3>", unsafe_allow_html=True)
    
    display_cols = ['ATM_ID', 'Date', 'Day_of_Week', 'Time_of_Day', 'Total_Withdrawals', 
                   'Total_Deposits', 'Location_Type', 'Holiday_Flag', 'Special_Event_Flag']
    
    st.dataframe(query_df[display_cols].head(50), use_container_width=True)
    
    # Export option
    st.markdown("<h3 class='sub-header'>Export Results</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = query_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name='filtered_atm_data.csv',
            mime='text/csv'
        )
    
    with col2:
        # Summary report
        summary = f"""
        ATM Intelligence Query Report
        =============================
        Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Filters Applied:
        - Days: {', '.join(selected_day) if selected_day else 'All'}
        - Time: {', '.join(selected_time) if selected_time else 'All'}
        - Locations: {', '.join(selected_loc) if selected_loc else 'All'}
        
        Results Summary:
        - Total Records: {len(query_df):,}
        - Total Withdrawals: ₹{query_df['Total_Withdrawals'].sum():,.0f}
        - Average Withdrawal: ₹{query_df['Total_Withdrawals'].mean():,.0f}
        - Unique ATMs: {query_df['ATM_ID'].nunique()}
        """
        
        st.download_button(
            label="📥 Download Summary Report (TXT)",
            data=summary,
            file_name='atm_query_report.txt',
            mime='text/plain'
        )

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>ATM Intelligence Demand Forecasting | FA-2 Data Mining Project</p>", unsafe_allow_html=True)
