# =========================================
# FA-1 ATM Intelligence Demand Forecasting
# Data Preprocessing + Exploration
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries loaded")


# =========================================
# Load dataset
# =========================================

file_path = "atm_cash_management_dataset.csv"

try:
    df = pd.read_csv(file_path)
    print("Dataset loaded")
    print(df.head())
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please ensure the CSV file is in the same directory.")
    exit()


# =========================================
# Dataset info
# =========================================

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.info())


# =========================================
# Missing values
# =========================================

print("Missing values:")
print(df.isnull().sum())


# Fill missing values
if "Holiday_Flag" in df.columns:
    df["Holiday_Flag"] = df["Holiday_Flag"].fillna(0)
if "Special_Event_Flag" in df.columns:
    df["Special_Event_Flag"] = df["Special_Event_Flag"].fillna(0)

df = df.dropna()

print("After cleaning")
print(df.isnull().sum())


# =========================================
# Date formatting
# =========================================

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    
    df["Month"] = df["Date"].dt.month
    df["Week_Number"] = df["Date"].dt.isocalendar().week
    df["Day_Name"] = df["Date"].dt.day_name()
    
    print(df.head())


# =========================================
# Encoding categories
# =========================================

day_map = {
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
    "Sunday": 7
}

if "Day_Name" in df.columns:
    df["Day_of_Week"] = df["Day_Name"].map(day_map)


time_map = {
    "Morning": 1,
    "Afternoon": 2,
    "Evening": 3,
    "Night": 4
}

if "Time_of_Day" in df.columns:
    df["Time_of_Day"] = df["Time_of_Day"].map(time_map)


loc_map = {
    "Urban": 1,
    "Semi-Urban": 2,
    "Rural": 3
}

if "Location_Type" in df.columns:
    df["Location_Type"] = df["Location_Type"].map(loc_map)


weather_map = {
    "Sunny": 1,
    "Rainy": 2,
    "Cloudy": 3,
    "Storm": 4
}

if "Weather_Condition" in df.columns:
    df["Weather_Condition"] = df["Weather_Condition"].map(weather_map)

print("Encoding done")


# =========================================
# Normalization
# =========================================

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

num_cols = [
    "Total_Withdrawals",
    "Total_Deposits",
    "Previous_Day_Cash_Level",
    "Cash_Demand_Next_Day"
]

# Only normalize columns that exist
existing_num_cols = [col for col in num_cols if col in df.columns]
if existing_num_cols:
    df[existing_num_cols] = scaler.fit_transform(df[existing_num_cols])
    print("Normalization done")
else:
    print("Warning: No numeric columns found for normalization")


# =========================================
# Logical check
# =========================================

if "Total_Withdrawals" in df.columns and "Previous_Day_Cash_Level" in df.columns:
    df["Error_Flag"] = df["Total_Withdrawals"] > df["Previous_Day_Cash_Level"]
    print("Errors found:", df["Error_Flag"].sum())


# =========================================
# VISUALIZATION (for storyboard)
# =========================================

# Use non-interactive backend for Streamlit Cloud compatibility
import matplotlib
matplotlib.use('Agg')

if "Total_Withdrawals" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Total_Withdrawals"])
    plt.title("Withdrawals Distribution")
    plt.tight_layout()
    plt.savefig("withdrawals_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df["Total_Withdrawals"])
    plt.title("Outliers Check")
    plt.tight_layout()
    plt.savefig("outliers_check.png")
    plt.close()

if "Date" in df.columns and "Total_Withdrawals" in df.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Total_Withdrawals"])
    plt.title("Withdrawals Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total Withdrawals")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("withdrawals_over_time.png")
    plt.close()

if "Holiday_Flag" in df.columns and "Total_Withdrawals" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df["Holiday_Flag"], y=df["Total_Withdrawals"])
    plt.title("Holiday Impact")
    plt.tight_layout()
    plt.savefig("holiday_impact.png")
    plt.close()

if "Day_of_Week" in df.columns and "Total_Withdrawals" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df["Day_of_Week"], y=df["Total_Withdrawals"])
    plt.title("Withdrawals by Day")
    plt.tight_layout()
    plt.savefig("withdrawals_by_day.png")
    plt.close()

if "Previous_Day_Cash_Level" in df.columns and "Cash_Demand_Next_Day" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=df["Previous_Day_Cash_Level"],
        y=df["Cash_Demand_Next_Day"]
    )
    plt.title("Cash Level vs Next Day Demand")
    plt.tight_layout()
    plt.savefig("cash_level_vs_demand.png")
    plt.close()

print("Visualizations saved as PNG files")


# =========================================
# Save cleaned dataset
# =========================================

df.to_csv("cleaned_atm_data.csv", index=False)

print("Saved cleaned data")
print("Process completed successfully!")
