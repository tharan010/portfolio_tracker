# dashboard.py

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta

# ─── Import existing loader/analytics routines ───────────────────────────
from load_and_analyze import (
    build_portfolio_with_current_prices,
    compute_portfolio_weights,
    get_historical_data,
    compute_max_drawdown_from_df,
    compute_historical_volatility_from_df,
    compute_beta_from_df,
    compute_cost_of_equity_from_df,
    get_sector_allocation
)

# ──────────────────────────────────────────────────────────────────────────────
# 1) STREAMLIT PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📊 Portfolio Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Portfolio Analytics Dashboard")
st.markdown(
    """
    Upload **`portfolio_template.xlsx`** to explore:
    1. Current Metrics (Invested / Net Worth / P/L)  
    2. Asset‐Class & Instrument‐Type Allocation (side by side)  
    3. Historical Return Correlations (larger heatmap)  
    4. Max Drawdown & Volatility (larger bar charts)  
    """
)

# ──────────────────────────────────────────────────────────────────────────────
# 2) SIDEBAR: FILE UPLOADER & SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Settings")
    uploaded_file = st.file_uploader(
        label="Upload Portfolio Excel (.xlsx)",
        type=["xlsx"]
    )

    lookback_years = st.slider(
        label="Historical Lookback (years)",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of years back to fetch historical prices/NAVs"
    )

    st.markdown("---")
    refresh_button = st.button("🔄 Refresh Data")

# ──────────────────────────────────────────────────────────────────────────────
# 3) LOAD + CACHE PORTFOLIO DATA
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_prepare_portfolio(file_buffer, lookback_years: int):
    """
    1. Build portfolio from Excel and fetch current prices (yfinance + mftool)
    2. Compute asset‐weight & class‐allocation
    3. Download historical prices/NAVs over the past `lookback_years` years
    4. Calculate beta ratios and cost of equity
    5. Get sector allocation
    Returns:
      df_portfolio, asset_weights_df, class_alloc_df, combined_hist_df,
      beta_series, cost_of_equity_series, instrument_sector_df, sector_allocation_df
    """
    # 1.a) Build current‐value portfolio
    df = build_portfolio_with_current_prices(file_buffer)

    # 1.b) Compute weights & class allocation
    asset_weights_df, class_alloc_df = compute_portfolio_weights(df)

    # 1.c) Fetch historical prices/NAVs in one shot
    combined_hist_df = get_historical_data(df, lookback_years=lookback_years)

    beta_series = compute_beta_from_df(combined_hist_df, df, lookback_years)
    cost_of_equity_series = compute_cost_of_equity_from_df(combined_hist_df, df, lookback_years)
    
    # NEW: Get sector allocation
    instrument_sector_df, sector_allocation_df = get_sector_allocation(df)

    return (df, asset_weights_df, class_alloc_df, combined_hist_df, 
            beta_series, cost_of_equity_series, instrument_sector_df, sector_allocation_df)
    # return df, asset_weights_df, class_alloc_df, combined_hist_df

# If no file is uploaded, show a placeholder
if uploaded_file is None:
    st.warning("📁 Please upload your `portfolio_template.xlsx` in the sidebar.")
    st.stop()

# Load (and cache) everything at once
(df_portfolio, asset_weights, class_alloc, combined_hist, 
 beta_data, cost_of_equity_data, instrument_sectors, sector_alloc) = load_and_prepare_portfolio(
    uploaded_file, lookback_years
)

# ──────────────────────────────────────────────────────────────────────────────
# 4) DISPLAY CURRENT‐VALUE METRICS
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("💡 Portfolio Summary")
col1, col2, col3 = st.columns(3)

# Recompute Total Invested, Net Worth, P/L
total_invested = (
    df_portfolio["Quantity"]
    * df_portfolio["Purchase Price"]
    * np.where(df_portfolio["Currency"] == "USD", 85, 1)
).sum()

net_worth = df_portfolio["Current Value INR"].sum()
profit_pct = (net_worth - total_invested) / total_invested * 100

with col1:
    st.metric("Total Invested (INR)", f"₹{total_invested:,.2f}")
with col2:
    st.metric("Net Worth (INR)", f"₹{net_worth:,.2f}")
with col3:
    st.metric("Overall P/L (%)", f"{profit_pct:.2f}%")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# 5 & 6) ASSET‐CLASS & INSTRUMENT‐TYPE ALLOCATION SIDE BY SIDE
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("📊 Allocation by Instrument Type & Breakdown by Instrument Type")
colA, colB = st.columns(2)

# Left: Donut‐style pie chart for Instrument Type allocation
with colA:
    class_alloc_copy = class_alloc.copy()
    class_alloc_copy["Pct"] = (class_alloc_copy["Total Value INR"] / net_worth * 100).round(2)

    fig_alloc = px.pie(
        class_alloc_copy,
        names="Instrument Type",
        values="Pct",
        title="Instrument‐Type Allocation (%)",
        hole=0.5
    )
    fig_alloc.update_traces(
        textinfo="label+percent",
        textfont_size=16,
        marker=dict(line=dict(color="white", width=2))
    )
    fig_alloc.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        width=500,
        height=500,
        font=dict(size=14)
    )
    st.plotly_chart(fig_alloc, use_container_width=True)

# Right: Horizontal bar chart for Instrument Type breakdown
with colB:
    inst_df = (
        df_portfolio
        .groupby("Instrument Type")["Current Value INR"]
        .sum()
        .reset_index(name="Total Value INR")
        .sort_values("Total Value INR", ascending=True)
    )

    fig_inst = px.bar(
        inst_df,
        x="Total Value INR",
        y="Instrument Type",
        orientation="h",
        text=inst_df["Total Value INR"].apply(lambda x: f"₹{x:,.0f}"),
        title="Current Value (INR) by Instrument Type",
        labels={"Total Value INR": "Current Value (₹)", "Instrument Type": ""}
    )
    fig_inst.update_traces(textfont_size=20)
    fig_inst.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        width=500,
        height=500,
        font=dict(size=20)
    )
    st.plotly_chart(fig_inst, use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# 7) RETURN CORRELATION (HEATMAP)
# ──────────────────────────────────────────────────────────────────────────────
st.subheader(f"🔗 Return Correlation (Last {lookback_years} Years)")

numeric_hist = combined_hist.apply(pd.to_numeric, errors="coerce")
daily_returns = numeric_hist.pct_change(fill_method=None).dropna(how="all")

if daily_returns.shape[1] > 1:
    corr_matrix = daily_returns.corr()

    fig_corr = px.imshow(
        corr_matrix,
        labels=dict(x="Instrument", y="Instrument", color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Return Correlation Matrix"
    )
    fig_corr.update_traces(textfont_size=20)
    fig_corr.update_layout(
        xaxis_tickangle=-90,
        margin=dict(l=20, r=20, t=60, b=20),
        width=900,
        height=900,
        font=dict(size=40),
        title_font_size=24
    )
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.warning("Not enough instruments to compute a correlation matrix.")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# 8) MAX DRAWDOWN & 9) HISTORICAL VOLATILITY (LARGER BAR CHARTS)
# ──────────────────────────────────────────────────────────────────────────────
colC, colD = st.columns(2)

# Left: Maximum Drawdown
with colC:
    st.subheader(f"📉 Maximum Drawdown (Last {lookback_years} Years)")
    dd_series = compute_max_drawdown_from_df(combined_hist, df_portfolio)
    dd_pct = (dd_series * 100).sort_values()

    if not dd_pct.empty:
        dd_df = dd_pct.reset_index()
        dd_df.columns = ["Instrument", "Max Drawdown (%)"]

        fig_dd = px.bar(
            dd_df,
            x="Max Drawdown (%)",
            y="Instrument",
            orientation="h",
            text=dd_df["Max Drawdown (%)"].apply(lambda x: f"{x:.1f}%"),
            title="Maximum Drawdown by Instrument",
            color="Max Drawdown (%)",
            color_continuous_scale="Reds"
        )
        fig_dd.update_traces(textfont_size=24)
        fig_dd.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            width=600,
            height=600,
            font=dict(size=20),
            title_font_size=24
        )
        st.plotly_chart(fig_dd, use_container_width=True)
    else:
        st.warning("No drawdown data available.")

# Right: Historical Volatility
with colD:
    st.subheader(f"📊 Annualized Volatility (Last {lookback_years} Years)")
    vol_series = compute_historical_volatility_from_df(combined_hist, df_portfolio)
    vol_pct = (vol_series * 100).sort_values()

    if not vol_pct.empty:
        vol_df = vol_pct.reset_index()
        vol_df.columns = ["Instrument", "Annualized Volatility (%)"]

        fig_vol = px.bar(
            vol_df,
            x="Annualized Volatility (%)",
            y="Instrument",
            orientation="h",
            text=vol_df["Annualized Volatility (%)"].apply(lambda x: f"{x:.1f}%"),
            title="Annualized Volatility by Instrument",
            color="Annualized Volatility (%)",
            color_continuous_scale="Blues"
        )
        fig_vol.update_traces(textfont_size=24)
        fig_vol.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            width=600,
            height=600,
            font=dict(size=20),
            title_font_size=24
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.warning("No volatility data available.")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# 10) FULL PORTFOLIO TABLE
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("📋 Full Portfolio Details")
sorted_df = df_portfolio.sort_values(by="Current Value INR", ascending=False).fillna("")
st.dataframe(sorted_df, use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# 11) BETA RATIO & COST OF EQUITY VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────
st.subheader(f"📈 Risk Metrics for Equity Instruments (Last {lookback_years} Years)")
colE, colF = st.columns(2)

# Left: Beta Ratio
with colE:
    st.subheader("β Beta Ratio")
    if not beta_data.empty and not beta_data.isna().all():
        beta_clean = beta_data.dropna().sort_values()
        
        if not beta_clean.empty:
            beta_df = beta_clean.reset_index()
            beta_df.columns = ["Instrument", "Beta"]

            fig_beta = px.bar(
                beta_df,
                x="Beta",
                y="Instrument", 
                orientation="h",
                text=beta_df["Beta"].apply(lambda x: f"{x:.2f}"),
                title="Beta Ratios (vs SPY for USD, NIFTY for INR)",
                color="Beta",
                color_continuous_scale="RdYlBu_r"
            )
            fig_beta.update_traces(textfont_size=20)
            fig_beta.update_layout(
                margin=dict(l=10, r=10, t=50, b=10),
                width=600,
                height=500,
                font=dict(size=16),
                title_font_size=20
            )
            # Add reference line at Beta = 1
            fig_beta.add_vline(x=1, line_dash="dash", line_color="black", 
                              annotation_text="Market Beta = 1")
            st.plotly_chart(fig_beta, use_container_width=True)
        else:
            st.warning("No beta data available for equity instruments.")
    else:
        st.warning("No beta data available.")

# Right: Cost of Equity
with colF:
    st.subheader("💰 Cost of Equity (CAPM)")
    if not cost_of_equity_data.empty and not cost_of_equity_data.isna().all():
        coe_clean = (cost_of_equity_data * 100).dropna().sort_values()  # Convert to percentage
        
        if not coe_clean.empty:
            coe_df = coe_clean.reset_index()
            coe_df.columns = ["Instrument", "Cost of Equity (%)"]

            fig_coe = px.bar(
                coe_df,
                x="Cost of Equity (%)",
                y="Instrument",
                orientation="h", 
                text=coe_df["Cost of Equity (%)"].apply(lambda x: f"{x:.1f}%"),
                title="Cost of Equity using CAPM Model",
                color="Cost of Equity (%)",
                color_continuous_scale="Reds"
            )
            fig_coe.update_traces(textfont_size=20)
            fig_coe.update_layout(
                margin=dict(l=10, r=10, t=50, b=10),
                width=600,
                height=500,
                font=dict(size=16),
                title_font_size=20
            )
            st.plotly_chart(fig_coe, use_container_width=True)
        else:
            st.warning("No cost of equity data available.")
    else:
        st.warning("No cost of equity data available.")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# 12) SECTOR ALLOCATION VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("🏭 Sector Allocation (Equity Instruments Only)")

if not sector_alloc.empty:
    colG, colH = st.columns(2)
    
    # Left: Sector allocation pie chart
    with colG:
        fig_sector_pie = px.pie(
            sector_alloc,
            names="Sector",
            values="Weight (%)",
            title="Sector Allocation (%)",
            hole=0.4
        )
        fig_sector_pie.update_traces(
            textinfo="label+percent",
            textfont_size=14,
            marker=dict(line=dict(color="white", width=2))
        )
        fig_sector_pie.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            width=500,
            height=500,
            font=dict(size=14)
        )
        st.plotly_chart(fig_sector_pie, use_container_width=True)
    
    # Right: Sector allocation bar chart
    with colH:
        fig_sector_bar = px.bar(
            sector_alloc.sort_values("Total Value INR", ascending=True),
            x="Total Value INR",
            y="Sector",
            orientation="h",
            text=sector_alloc.sort_values("Total Value INR", ascending=True)["Total Value INR"].apply(lambda x: f"₹{x:,.0f}"),
            title="Sector Allocation (INR Value)",
            color="Total Value INR",
            color_continuous_scale="Viridis"
        )
        fig_sector_bar.update_traces(textfont_size=16)
        fig_sector_bar.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            width=500,
            height=500,
            font=dict(size=14)
        )
        st.plotly_chart(fig_sector_bar, use_container_width=True)
    
    # Sector allocation table
    st.subheader("📊 Detailed Sector Breakdown")
    st.dataframe(sector_alloc, use_container_width=True)
    
else:
    st.warning("No equity instruments found for sector allocation analysis.")

# ──────────────────────────────────────────────────────────────────────────────
# 13) RISK METRICS TABLE
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Risk Metrics Summary (Equity Instruments)")

# Create a comprehensive risk metrics table
if not beta_data.empty or not cost_of_equity_data.empty:
    risk_metrics_df = pd.DataFrame(index=beta_data.index.union(cost_of_equity_data.index))
    
    if not beta_data.empty:
        risk_metrics_df['Beta'] = beta_data
    if not cost_of_equity_data.empty:
        risk_metrics_df['Cost of Equity (%)'] = (cost_of_equity_data * 100).round(2)
    
    # Add sector information if available
    if not instrument_sectors.empty:
        sector_map = instrument_sectors.set_index('Name')['Sector'].to_dict()
        risk_metrics_df['Sector'] = risk_metrics_df.index.map(sector_map)
    
    # Add current values for context
    value_map = df_portfolio.set_index('Name')['Current Value INR'].to_dict()
    risk_metrics_df['Current Value INR'] = risk_metrics_df.index.map(value_map)
    
    # Sort by current value
    risk_metrics_df = risk_metrics_df.sort_values('Current Value INR', ascending=False, na_position='last')
    
    # Clean up the dataframe
    risk_metrics_df = risk_metrics_df.round(2).fillna('-')
    
    st.dataframe(risk_metrics_df, use_container_width=True)
else:
    st.warning("No risk metrics data available to display.")
# End of dashboard