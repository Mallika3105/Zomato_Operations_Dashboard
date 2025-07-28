# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("zomato_dibba_demo_dataset.csv")
df_full = df.copy()  # Keep a copy of the full dataset before applying filters
st.set_page_config(page_title="Zomato Dibba Ops Dashboard", layout="wide")

st.title("📊 Zomato Dibba Operations Dashboard")

# Sidebar filters
st.sidebar.header("Filter by:")
selected_city = st.sidebar.multiselect("City", sorted(df['City'].unique()), default=None)
selected_cuisine = st.sidebar.multiselect("Preferred Cuisine", sorted(df['Preferred_Cuisine'].unique()), default=None)

# Filter logic
filtered_df = df.copy()
if selected_city:
    filtered_df = filtered_df[filtered_df['City'].isin(selected_city)]
if selected_cuisine:
    filtered_df = filtered_df[filtered_df['Preferred_Cuisine'].isin(selected_cuisine)]

# ✅ Safe KPI Cards to handle empty filtered_df or NaNs
col1, col2, col3, col4 = st.columns(4)

# Total Subscribers
if not filtered_df.empty:
    col1.metric("👥 Total Subscribers", f"{filtered_df['Subscribers'].sum():,}")
else:
    col1.metric("👥 Total Subscribers", "N/A")

# Avg Meals/Day
if not filtered_df.empty and pd.notna(filtered_df['Avg_Meals_Delivered/Day'].mean()):
    avg_meals = filtered_df['Avg_Meals_Delivered/Day'].mean()
    col2.metric("🍱 Avg Meals/Day", f"{int(avg_meals):,}")
else:
    col2.metric("🍱 Avg Meals/Day", "N/A")
    st.warning("⚠️ No data available for this city and cuisine combination.")

# Avg Delivery Time
if not filtered_df.empty and pd.notna(filtered_df['Avg_Delivery_Time_Min'].mean()):
    col3.metric("⏱️ Avg Delivery Time (min)", f"{filtered_df['Avg_Delivery_Time_Min'].mean():.1f}")
else:
    col3.metric("⏱️ Avg Delivery Time (min)", "N/A")

# Daily Revenue
if not filtered_df.empty:
    col4.metric("💰 Daily Revenue", f"₹ {filtered_df['Daily_Revenue_₹'].sum():,}")
else:
    col4.metric("💰 Daily Revenue", "N/A")

# Data Table
st.subheader("📍 Zone-wise Operational Overview")
st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

# Highlight alerts
# 🔎 Risk Scoring & Attention
st.subheader("🚨 Zones Requiring Attention")

# Assign 1 point for each weak metric
filtered_df['Risk_Score'] = (
    (filtered_df['Avg_Kitchen_Rating'] < 3.5).astype(int) +
    (filtered_df['Ingredient_Fulfilment_%'] < 85).astype(int) +
    (filtered_df['Avg_Delivery_Time_Min'] > 35).astype(int)
)

# Zones requiring attention (moderate risk: 2 out of 3 fail)
alert_df = filtered_df[filtered_df['Risk_Score'] >= 2]

if not alert_df.empty:
    st.warning("The following zones may need intervention based on risk score (2+ red flags):")
    st.dataframe(alert_df.reset_index(drop=True), use_container_width=True)
else:
    st.success("✅ All zones performing well with low risk.")

import plotly.express as px

# Load time-series dataset
df_ts = pd.read_csv("zomato_dibba_timeseries.csv", parse_dates=["Date"])

st.header("📈 Zone-wise Time Series Analysis")

# Zone selection dropdown
if selected_city:
    df_ts_filtered = df_ts.merge(df_full[['Zone', 'City']], on='Zone')
    df_ts_filtered = df_ts_filtered[df_ts_filtered['City'].isin(selected_city)]
    zone_list = df_ts_filtered['Zone'].unique()
else:
    df_ts_filtered = df_ts
    zone_list = df_ts['Zone'].unique()

selected_zone = st.selectbox("Select a Zone to View 30-Day Trends:", sorted(zone_list))

# Filter and process zone data
zone_data = df_ts_filtered[df_ts_filtered['Zone'] == selected_zone].sort_values("Date")
zone_data['Meals_7DayAvg'] = zone_data['Meals_Delivered'].rolling(window=7).mean()
zone_data['Revenue_7DayAvg'] = zone_data['Revenue_₹'].rolling(window=7).mean()

# Meals chart
st.subheader(f"🍽️ Meals Delivered Trend - {selected_zone}")
fig_meals = px.line(zone_data, x='Date', y=['Meals_Delivered', 'Meals_7DayAvg'],
                    labels={"value": "Meals", "variable": "Legend"},
                    title="Meals Delivered (Daily vs 7-Day Avg)",
                    markers=True)
st.plotly_chart(fig_meals, use_container_width=True)

# Revenue chart
st.subheader(f"💸 Revenue Trend - {selected_zone}")
fig_revenue = px.line(zone_data, x='Date', y=['Revenue_₹', 'Revenue_7DayAvg'],
                      labels={"value": "Revenue (₹)", "variable": "Legend"},
                      title="Revenue (Daily vs 7-Day Avg)",
                      markers=True)
st.plotly_chart(fig_revenue, use_container_width=True)

# Merge city information into the time-series dataset
df_ts = pd.read_csv("zomato_dibba_timeseries.csv", parse_dates=["Date"])
df_full = pd.read_csv("zomato_dibba_demo_dataset.csv")  # In case not already loaded

df_ts_with_city = df_ts.merge(df_full[['Zone', 'City']], on='Zone')

if selected_city:
    df_ts_with_city = df_ts_with_city[df_ts_with_city['City'].isin(selected_city)]

df_city_timeseries = df_ts_with_city.groupby(['City', 'Date']).agg({
    'Meals_Delivered': 'sum',
    'Revenue_₹': 'sum'
}).reset_index()

st.header("🏙️ City-Level Aggregated Trends")

# Load merged and grouped data
df_city = df_city_timeseries.copy()

# Dropdown for city selection
selected_city_ts = st.selectbox("Select a City to View Aggregated 30-Day Trends:", df_city['City'].unique())

# Filter and sort data
city_data = df_city[df_city['City'] == selected_city_ts].sort_values("Date")
city_data['Meals_7DayAvg'] = city_data['Meals_Delivered'].rolling(window=7).mean()
city_data['Revenue_7DayAvg'] = city_data['Revenue_₹'].rolling(window=7).mean()

# Plot Meals
st.subheader(f"📦 Total Meals Delivered - {selected_city_ts}")
fig_city_meals = px.line(city_data, x='Date', y=['Meals_Delivered', 'Meals_7DayAvg'],
                         labels={"value": "Meals", "variable": "Legend"},
                         title=f"Meals Delivered (Daily vs 7-Day Avg) in {selected_city_ts}",
                         markers=True)
st.plotly_chart(fig_city_meals, use_container_width=True)

# Plot Revenue
st.subheader(f"💰 Total Revenue - {selected_city_ts}")
fig_city_revenue = px.line(city_data, x='Date', y=['Revenue_₹', 'Revenue_7DayAvg'],
                           labels={"value": "Revenue (₹)", "variable": "Legend"},
                           title=f"Revenue (Daily vs 7-Day Avg) in {selected_city_ts}",
                           markers=True)
st.plotly_chart(fig_city_revenue, use_container_width=True)

from prophet import Prophet
from prophet.plot import plot_plotly

st.header("🔮 Forecast with Prophet (15-Day Horizon)")

# Select city for forecasting
forecast_city = st.selectbox("Select a City for Forecasting:", df_city_timeseries['City'].unique(), key='forecast_city')

# Prepare data for Prophet
city_df = df_city_timeseries[df_city_timeseries['City'] == forecast_city][['Date', 'Meals_Delivered', 'Revenue_₹']].copy()
city_df = city_df.sort_values('Date')

# Forecast Meals Delivered
st.subheader(f"🍽️ Meals Forecast for {forecast_city}")

meals_df = city_df.rename(columns={'Date': 'ds', 'Meals_Delivered': 'y'})
meals_model = Prophet()
meals_model.fit(meals_df)

future_meals = meals_model.make_future_dataframe(periods=15)
forecast_meals = meals_model.predict(future_meals)

fig_meals = plot_plotly(meals_model, forecast_meals)
st.plotly_chart(fig_meals, use_container_width=True)

# Forecast Revenue
st.subheader(f"💸 Revenue Forecast for {forecast_city}")

revenue_df = city_df.rename(columns={'Date': 'ds', 'Revenue_₹': 'y'})
revenue_model = Prophet()
revenue_model.fit(revenue_df)

future_revenue = revenue_model.make_future_dataframe(periods=15)
forecast_revenue = revenue_model.predict(future_revenue)

fig_revenue = plot_plotly(revenue_model, forecast_revenue)
st.plotly_chart(fig_revenue, use_container_width=True)

# 📦 Inventory Requirement Estimation (Pulse Forecast Integration)
st.subheader("📦 Estimated Inventory Requirement for Next 7 Days")

# 🧂 Define a complete ingredient map for all valid cuisines
ingredient_map = {
    'North Indian': {
        'Rice (g)': 200,
        'Dal (g)': 150,
        'Paneer (g)': 100
    },
    'South Indian': {
        'Rice (g)': 250,
        'Sambar (g)': 200,
        'Coconut (g)': 50
    },
    'Healthy': {
        'Quinoa (g)': 150,
        'Sprouts (g)': 100,
        'Boiled Veggies (g)': 150
    },
    'Mixed': {
        'Rice (g)': 150,
        'Dal (g)': 100,
        'Salad (g)': 100,
        'Curry (g)': 120
    },
    'Regional': {
        'Bajra (g)': 150,
        'Kadhi (g)': 120,
        'Sabzi (g)': 130
    }
}

# 👁 Get cuisine based on forecast city
city_cuisine_row = filtered_df[filtered_df['City'] == forecast_city]
if not city_cuisine_row.empty:
    cuisine = city_cuisine_row['Preferred_Cuisine'].iloc[0]

    if cuisine in ingredient_map:
        ingredients = ingredient_map[cuisine]

        # ✅ Check if forecast_meals exists and has 'ds' and 'yhat'
        if 'ds' in forecast_meals.columns and 'yhat' in forecast_meals.columns:
            forecast_7days = forecast_meals.tail(7)

            inventory_records = []
            for _, row in forecast_7days.iterrows():
                for ingredient, qty_per_meal in ingredients.items():
                    inventory_records.append({
                        'Date': row['ds'].date(),
                        'City': forecast_city,
                        'Cuisine': cuisine,
                        'Ingredient': ingredient,
                        'Estimated Quantity Required': int(row['yhat'] * qty_per_meal)
                    })

            inventory_df = pd.DataFrame(inventory_records)
            st.dataframe(inventory_df, use_container_width=True)
        else:
            st.warning("Forecast data not available or not properly formatted.")
    else:
        st.warning(f"⚠️ Ingredient mapping not found for selected cuisine: {cuisine}")
else:
    st.warning("⚠️ No matching city and cuisine found in the filtered data.")

# Generate weekday and week info from the city-level time series
df_city_timeseries['Day'] = df_city_timeseries['Date'].dt.day_name()
df_city_timeseries['Week'] = df_city_timeseries['Date'].dt.isocalendar().week

# Daily averages by weekday
daily_trend = df_city_timeseries.groupby('Day').agg({
    'Meals_Delivered': 'mean',
    'Revenue_₹': 'mean'
}).reset_index()

# Weekly totals
weekly_trend = df_city_timeseries.groupby('Week').agg({
    'Meals_Delivered': 'sum',
    'Revenue_₹': 'sum'
}).reset_index()

# Ensure correct weekday order
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_trend['Day'] = pd.Categorical(daily_trend['Day'], categories=day_order, ordered=True)
daily_trend = daily_trend.sort_values('Day')


# 📍 Profitability Ranking - Top 10 Most Profitable Zones

st.subheader("💸 Top 10 Most Profitable Zones")
cost_per_meal = 15 + 12 + 8

# Clean + calculate profitability
profit_data = filtered_df.copy()
profit_data['Estimated_Cost_₹'] = profit_data['Avg_Meals_Delivered/Day'] * cost_per_meal
profit_data['Profit_₹'] = profit_data['Daily_Revenue_₹'] - profit_data['Estimated_Cost_₹']
profit_data['Profit_Margin_%'] = (profit_data['Profit_₹'] / profit_data['Daily_Revenue_₹'].replace(0, np.nan)) * 100

# Filter top profitable zones
top_profit_zones = profit_data.sort_values("Profit_₹", ascending=False).head(10)

# Bar chart
fig_bar = px.bar(
    top_profit_zones,
    x='Profit_₹',
    y='Zone',
    color='Profit_Margin_%',
    color_continuous_scale='RdYlGn',
    orientation='h',
    hover_data=['City', 'Subscribers', 'Avg_Kitchen_Rating', 'Ingredient_Fulfilment_%'],
    title="💰 Highest Profit-Generating Zones"
)

fig_bar.update_layout(
    xaxis_title="Profit (₹)",
    yaxis_title="Zone",
    yaxis=dict(autorange="reversed")
)

st.plotly_chart(fig_bar, use_container_width=True)

# Part B: Churn Risk Detector
st.subheader("⚠️ High Churn Risk Zones")

# High churn risk zones (all 3 red flags)
churn_risk_df = filtered_df[filtered_df['Risk_Score'] == 3]

if not churn_risk_df.empty:
    churn_risk_df['Risk_Flag'] = "⚠️ High"
    churn_risk_df = churn_risk_df[['City', 'Zone', 'Avg_Kitchen_Rating', 'Ingredient_Fulfilment_%',
                                   'Avg_Delivery_Time_Min', 'Subscribers', 'Daily_Revenue_₹', 'Risk_Flag']]
    st.warning("These zones show ALL 3 risk factors and need urgent attention:")
    st.dataframe(churn_risk_df.reset_index(drop=True), use_container_width=True)
else:
    st.info("🎯 No zones meet *all* 3 red-flag conditions. (Demo fallback below)")
    
    # Show top 2 risky zones as demo fallback
    fallback_df = filtered_df.sort_values("Risk_Score", ascending=False).head(2).copy()
    fallback_df['Risk_Flag'] = "⚠️ Demo"
    fallback_df = fallback_df[['City', 'Zone', 'Avg_Kitchen_Rating', 'Ingredient_Fulfilment_%',
                               'Avg_Delivery_Time_Min', 'Subscribers', 'Daily_Revenue_₹', 'Risk_Flag']]
    
    st.dataframe(fallback_df.reset_index(drop=True), use_container_width=True)


