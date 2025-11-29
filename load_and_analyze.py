import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from mftool import Mftool
import requests
mf = Mftool()
# from curl_cffi import requests
# session = requests.Session(impersonate="chrome")
# Commenting out curl_cffi session due to yfinance compatibility issues

# Load portfolio from Excel
def load_portfolio_from_excel(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath, engine="openpyxl")
    # Ensure numeric types for Quantity & Purchase Price
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["Purchase Price"] = pd.to_numeric(df["Purchase Price"], errors="coerce").fillna(0)
    # Force string type for Instrument Type, Name, Currency
    df["Instrument Type"] = df["Instrument Type"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Currency"] = df["Currency"].astype(str).str.strip().str.upper()
    return df

# 2) Fetch current prices for equities/ETFs/commodities via multiple methods
def get_current_price(symbols):
    if not symbols:
        return pd.Series(dtype=float)
    
    prices = {}
    
    for symbol in symbols:
        price = get_single_price(symbol)
        prices[symbol] = price
    
    return pd.Series(prices)

def get_single_price(symbol):
    """Get price for a single symbol using multiple fallback methods"""
    
    # Try ticker.fast_info (fastest and most reliable for current price)
    try:
        ticker = yf.Ticker(symbol)
        fast_info = ticker.fast_info
        if hasattr(fast_info, 'last_price') and fast_info.last_price is not None:
            return float(fast_info.last_price)
    except Exception as e:
        pass
    
    # Try ticker.info (slower but more comprehensive)
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if info and 'currentPrice' in info and info['currentPrice'] is not None:
            return float(info['currentPrice'])
        elif info and 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
            return float(info['regularMarketPrice'])
        elif info and 'previousClose' in info and info['previousClose'] is not None:
            return float(info['previousClose'])
    except Exception as e:
        pass
    
    # Try ticker.history (historical data approach)
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d", auto_adjust=True)
        if not hist.empty and 'Close' in hist.columns:
            # Get the most recent non-null close price
            close_prices = hist['Close'].dropna()
            if len(close_prices) > 0:
                return float(close_prices.iloc[-1])
    except Exception as e:
        pass
    
    # Try yf.download as last resort
    try:
        data = yf.download(symbol, period="5d", progress=False, auto_adjust=True)
        if not data.empty and 'Close' in data.columns:
            close_prices = data['Close'].dropna()
            if len(close_prices) > 0:
                return float(close_prices.iloc[-1])
    except Exception as e:
        pass
    
    # Try without auto_adjust
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d", auto_adjust=False)
        if not hist.empty and 'Close' in hist.columns:
            close_prices = hist['Close'].dropna()
            if len(close_prices) > 0:
                return float(close_prices.iloc[-1])
    except Exception as e:
        pass
    
    print(f"All methods failed for {symbol} - possibly delisted or invalid symbol")
    return np.nan

# 3) Fetch NAVs for mutual funds 
def get_mf_nav(symbols):
    nav_values = {}
    for symbol in symbols:
        nav = mf.calculate_balance_units_value(symbol, 10)  # adjust as needed
        nav_values[symbol] = float(nav['nav'])
    return pd.Series(nav_values)

# 4) Main routine to load and annotate portfolio
def build_portfolio_with_current_prices(filepath: str) -> pd.DataFrame:
    # 4a) Load
    portfolio = load_portfolio_from_excel(filepath)

    # 4b) Split out symbols by instrument type
    equity_symbols = portfolio.loc[portfolio['Instrument Type'] == 'Equity', 'Name'].tolist()
    etf_symbols    = portfolio.loc[portfolio['Instrument Type'] == 'ETF', 'Name'].tolist()
    com_symbols    = portfolio.loc[portfolio['Instrument Type'] == 'Commodity', 'Name'].tolist()
    mf_symbols     = portfolio.loc[portfolio['Instrument Type'] == 'Mutual Fund', 'Name'].tolist()

    # 4c) Fetch prices/NAVs
    equity_prices = get_current_price(equity_symbols)
    etf_prices    = get_current_price(etf_symbols)
    com_prices    = get_current_price(com_symbols)
    mf_navs       = get_mf_nav(mf_symbols)

    # 4d) Concatenate into a single Series: index = symbol, value = current price
    current_prices = pd.concat([equity_prices, etf_prices, com_prices, mf_navs])

    # 4e) Map back to portfolio DataFrame
    portfolio['Current Price'] = portfolio['Name'].map(current_prices)

    # -------------------------------------------------------------------
    # 4f) Override for cash/CD/Post Office, etc.
    fixed_price_types = ['Cash', 'CD', 'Post Office']  # must match your exact Instrument Type strings
    mask_fixed = portfolio['Instrument Type'].isin(fixed_price_types)
    portfolio.loc[mask_fixed, 'Current Price'] = portfolio.loc[mask_fixed, 'Purchase Price']

    usd_multiplier = np.where(portfolio['Currency'] == 'USD', 85, 1)
    portfolio['Current Value INR'] = portfolio['Quantity'] * portfolio['Current Price'] * usd_multiplier

    # -------------------------------------------------------------------
    numeric_cols = portfolio.select_dtypes(include='number').columns
    portfolio[numeric_cols] = portfolio[numeric_cols].round(2)

    return portfolio

df_portfolio = build_portfolio_with_current_prices("portfolio_template.xlsx")


# PORTFOLIO ANALYTICS

# ─── 1) PORTFOLIO WEIGHTS ──────────────────────────────────────────────────────

def compute_portfolio_weights(df_portfolio: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given df_portfolio (with columns "Name", "Instrument Type", and "Current Value INR"),
    returns two DataFrames:
      1. asset_weights: each row = one asset (Name), its type, current value, and weight % of total.
      2. class_alloc: aggregated by Instrument Type, showing total value and weight %.
    """
    if 'Current Value INR' not in df_portfolio.columns:
        raise KeyError("'Current Value INR' column not found in df_portfolio")

    total_value = df_portfolio['Current Value INR'].sum()

    # Individual‐asset weights
    asset_weights = df_portfolio[['Name', 'Instrument Type', 'Current Value INR']].copy()
    asset_weights['Weight (%)'] = (asset_weights['Current Value INR'] / total_value * 100).round(2)

    # Asset‐class‐level aggregation
    class_alloc = (
        df_portfolio
        .groupby('Instrument Type')['Current Value INR']
        .sum()
        .reset_index(name='Total Value INR')
    )
    class_alloc['Weight (%)'] = (class_alloc['Total Value INR'] / total_value * 100).round(2)

    return asset_weights, class_alloc


# ─── 2) GET HISTORICAL DATA ─────────────────────────────────────────────────


def get_historical_data(df_portfolio: pd.DataFrame, lookback_years: int = 5) -> pd.DataFrame:
    """
    Downloads and returns a DataFrame 'combined' of historical prices/NAVs for all instruments
    in df_portfolio over the past `lookback_years` years (plus a 30-day buffer).
    """
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=365 * lookback_years + 30)

    # 1) Separate equities vs. mutual funds
    equities = (
        df_portfolio.loc[
            df_portfolio["Instrument Type"].str.lower().isin(
                ["equity", "stock", "etf", "commodity"]
            ),
            "Name"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    mfs = (
        df_portfolio.loc[
            df_portfolio["Instrument Type"].str.lower().isin(
                ["mf", "mutual fund"]
            ),
            "Name"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    # 2) Download equity prices via yfinance
    if equities:
        try:
            equity_data = yf.download(
                tickers=equities,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                group_by="ticker",
                progress=False,
                # session=session
            )

            if equity_data.empty:
                stock_df = pd.DataFrame()
            elif len(equities) == 1:
                # Single ticker case
                sym = equities[0]
                if isinstance(equity_data, pd.DataFrame) and "Close" in equity_data.columns:
                    stock_df = equity_data[["Close"]].rename(columns={"Close": sym})
                else:
                    stock_df = pd.DataFrame()
            else:
                # Multi-ticker case
                if hasattr(equity_data.columns, 'levels') and len(equity_data.columns.levels) == 2:
                    # MultiIndex columns case
                    close_columns = []
                    for sym in equities:
                        if (sym, 'Close') in equity_data.columns:
                            close_columns.append((sym, 'Close'))
                    
                    if close_columns:
                        stock_df = equity_data[close_columns].copy()
                        # Flatten column names
                        stock_df.columns = [col[0] for col in stock_df.columns]
                    else:
                        stock_df = pd.DataFrame()
                else:
                    stock_df = pd.DataFrame()

            # Ensure index is datetime and sorted
            if not stock_df.empty:
                stock_df.index = pd.to_datetime(stock_df.index)
                stock_df = stock_df.sort_index()
        
        except Exception as e:
            print(f"Error downloading equity data: {e}")
            stock_df = pd.DataFrame()
    else:
        stock_df = pd.DataFrame()

    # 3) Download MF NAVs via mftool
    if mfs:
        try:
            mf = Mftool() 
            nav_dict = {}
            for code in mfs:
                try:
                    hist = mf.get_scheme_historical_nav(code, as_Dataframe=True)
                    if not hist.empty:
                        # Convert index to datetime
                        hist.index = pd.to_datetime(hist.index, dayfirst=True)
                        nav_series = hist["nav"].rename(code).sort_index()
                        nav_dict[code] = nav_series
                except Exception as e:
                    print(f"Error getting historical data for MF {code}: {e}")

            if nav_dict:
                mf_df = pd.DataFrame(nav_dict)
                if mf_df.index.name != "Date":
                    mf_df.index.name = "Date"
                mf_df = mf_df.sort_index()
            else:
                mf_df = pd.DataFrame()
        except Exception as e:
            print(f"Error downloading MF data: {e}")
            mf_df = pd.DataFrame()
    else:
        mf_df = pd.DataFrame()

    # 4) Merge stock_df and mf_df (outer join on date)
    if not stock_df.empty and not mf_df.empty:
        combined = pd.merge(
            stock_df, mf_df,
            how="outer",
            left_index=True,
            right_index=True
        )
    elif not stock_df.empty:
        combined = stock_df.copy()
    elif not mf_df.empty:
        combined = mf_df.copy()
    else:
        combined = pd.DataFrame()

    # 5) Reindex to include every calendar day in the range
    if not combined.empty:
        full_index = pd.date_range(start=start_date, end=end_date, freq="D")
        combined = combined.reindex(full_index)
        combined.index.name = "Date"

    return combined


# ─── 3) PLOT RETURN CORRELATION ─────────────────────────────────────────────────

def plot_return_correlation_from_df(combined_df: pd.DataFrame) -> None:
    """
    Given a DataFrame `combined_df` of historical prices/NAVs (indexed by Date,
    with columns = tickers or scheme codes), compute daily returns, then plot a
    correlation heatmap.
    """
    numeric_df = combined_df.apply(pd.to_numeric, errors="coerce")
    returns = numeric_df.pct_change(fill_method=None).dropna(how="all")
    corr_matrix = returns.corr()

    plt.figure(figsize=(10, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        xticklabels=corr_matrix.columns,
        yticklabels=corr_matrix.columns
    )
    plt.title("Return Correlation Matrix (Equities & Mutual Funds)", fontsize=16)
    plt.xlabel("Instrument", fontsize=12)
    plt.ylabel("Instrument", fontsize=12)
    plt.tight_layout()
    plt.show()


# ─── 4) COMPUTE MAX DRAW DOWN ─────────────────────────────────────────────────

def compute_max_drawdown_from_df(combined_df: pd.DataFrame, df_portfolio: pd.DataFrame = None) -> pd.Series:
    """
    Given a DataFrame `combined_df` of historical prices/NAVs (indexed by Date,
    with columns = tickers or scheme codes), compute each column's maximum drawdown:
      max_drawdown = min_over_t [(price_t / running_max_t) - 1].
    Returns a Series indexed by instrument, with values = max drawdown (negative decimal).
    
    If df_portfolio is provided, includes Cash instruments with 0% drawdown.
    """
    numeric_df = combined_df.apply(pd.to_numeric, errors="coerce")
    max_dd = {}
    for col in numeric_df.columns:
        running_max = numeric_df[col].cummax()
        drawdown = numeric_df[col] / running_max - 1
        max_dd[col] = drawdown.min()
    
    # Add Cash instruments with 0% drawdown
    if df_portfolio is not None:
        cash_instruments = df_portfolio[
            df_portfolio['Instrument Type'].str.lower().isin(['cash', 'cd', 'post office'])
        ]['Name'].tolist()
        
        for cash_instrument in cash_instruments:
            if cash_instrument not in max_dd:
                max_dd[cash_instrument] = 0.0  # Cash has no drawdown
    
    return pd.Series(max_dd, name="Max Drawdown")


# ─── 5) COMPUTE HISTORICAL VOLATILITY ─────────────────────────────────────────

def compute_historical_volatility_from_df(combined_df: pd.DataFrame, df_portfolio: pd.DataFrame = None) -> pd.Series:
    """
    Given a DataFrame `combined_df` of historical prices/NAVs (indexed by Date,
    with columns = tickers or scheme codes), compute each column's annualized
    volatility: σ_annual = std(daily returns) * sqrt(252). Returns a Series
    indexed by instrument, with values = volatility (decimal).
    
    If df_portfolio is provided, includes Cash instruments with 0% volatility.
    """
    numeric_df = combined_df.apply(pd.to_numeric, errors="coerce")
    daily_returns = numeric_df.pct_change(fill_method=None).dropna(how="all")

    vol_dict = {}
    for col in daily_returns.columns:
        sigma_daily = daily_returns[col].std(ddof=1)      # sample std dev
        sigma_annual = sigma_daily * np.sqrt(252)         # annualize
        vol_dict[col] = sigma_annual

    # Add Cash instruments with 0% volatility
    if df_portfolio is not None:
        cash_instruments = df_portfolio[
            df_portfolio['Instrument Type'].str.lower().isin(['cash', 'cd', 'post office'])
        ]['Name'].tolist()
        
        for cash_instrument in cash_instruments:
            if cash_instrument not in vol_dict:
                vol_dict[cash_instrument] = 0.0  # Cash has no volatility

    return pd.Series(vol_dict, name="Annualized Volatility")

import yfinance as yf

# ─── 6) CALCULATE BETA RATIO ─────────────────────────────────────────────────

def compute_beta_from_df(combined_df: pd.DataFrame, df_portfolio: pd.DataFrame, lookback_years: int = 5) -> pd.Series:
    """
    Calculate beta for each equity instrument against appropriate benchmark:
    - USD currency stocks: SPY as benchmark  
    - INR currency stocks: ^NSEI (NIFTY 50) as benchmark
    
    Beta = Covariance(stock_returns, benchmark_returns) / Variance(benchmark_returns)
    """
    # Get benchmark data
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=365 * lookback_years + 30)
    
    try:
        # Download benchmark data
        spy_data = yf.download("SPY", start=start_date, end=end_date, auto_adjust=True, progress=False)
        nifty_data = yf.download("^NSEI", start=start_date, end=end_date, auto_adjust=True, progress=False)
        
        # Handle single column case for benchmarks
        if isinstance(spy_data, pd.DataFrame) and 'Close' in spy_data.columns:
            spy_returns = spy_data['Close'].pct_change(fill_method=None).dropna()
        else:
            spy_returns = pd.Series(dtype=float)
            
        if isinstance(nifty_data, pd.DataFrame) and 'Close' in nifty_data.columns:
            nifty_returns = nifty_data['Close'].pct_change(fill_method=None).dropna()
        else:
            nifty_returns = pd.Series(dtype=float)
            
    except Exception as e:
        print(f"Error downloading benchmark data: {e}")
        return pd.Series(dtype=float, name="Beta")
    
    # Calculate returns for portfolio instruments
    numeric_df = combined_df.apply(pd.to_numeric, errors="coerce")
    daily_returns = numeric_df.pct_change(fill_method=None).dropna(how="all")
    
    beta_dict = {}
    
    # Get equity instruments only
    equity_instruments = df_portfolio[
        df_portfolio["Instrument Type"].str.lower().isin(["equity", "stock", "etf"])
    ]
    
    for _, row in equity_instruments.iterrows():
        instrument = row["Name"]
        currency = row["Currency"]
        
        if instrument not in daily_returns.columns:
            beta_dict[instrument] = np.nan
            continue
            
        stock_returns = daily_returns[instrument].dropna()
        
        # Choose benchmark based on currency
        if currency == "USD":
            benchmark_returns = spy_returns
        else:  # INR or other
            benchmark_returns = nifty_returns
        
        # Check if benchmark data is available
        if benchmark_returns.empty:
            beta_dict[instrument] = np.nan
            continue
        
        # Align dates
        common_dates = stock_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) < 30:  # Need at least 30 data points
            beta_dict[instrument] = np.nan
            continue
            
        stock_aligned = stock_returns.loc[common_dates]
        benchmark_aligned = benchmark_returns.loc[common_dates]
        
        # Ensure we have Series, not DataFrames
        if isinstance(stock_aligned, pd.DataFrame):
            stock_aligned = stock_aligned.iloc[:, 0]  # Take first column
        if isinstance(benchmark_aligned, pd.DataFrame):
            benchmark_aligned = benchmark_aligned.iloc[:, 0]  # Take first column
            
        # Remove any remaining NaN values
        valid_indices = stock_aligned.notna() & benchmark_aligned.notna()
        stock_clean = stock_aligned[valid_indices]
        benchmark_clean = benchmark_aligned[valid_indices]
        
        if len(stock_clean) < 30:
            beta_dict[instrument] = np.nan
            continue
        
        try:
            # Calculate beta using correlation and standard deviations
            correlation = stock_clean.corr(benchmark_clean)
            stock_std = stock_clean.std()
            benchmark_std = benchmark_clean.std()
            
            if benchmark_std != 0 and not pd.isna(correlation):
                beta = correlation * (stock_std / benchmark_std)
                beta_dict[instrument] = beta
            else:
                beta_dict[instrument] = np.nan
                
        except Exception as e:
            print(f"Error calculating beta for {instrument}: {e}")
            beta_dict[instrument] = np.nan
    
    # Add Cash instruments with beta = 0 (risk-free)
    cash_instruments = df_portfolio[
        df_portfolio['Instrument Type'].str.lower().isin(['cash', 'cd', 'post office'])
    ]['Name'].tolist()
    
    for cash_instrument in cash_instruments:
        if cash_instrument not in beta_dict:
            beta_dict[cash_instrument] = 0.0  # Cash has no systematic risk (beta = 0)
    
    return pd.Series(beta_dict, name="Beta")


# ─── 7) CALCULATE COST OF EQUITY ─────────────────────────────────────────────

def compute_cost_of_equity_from_df(combined_df: pd.DataFrame, df_portfolio: pd.DataFrame, lookback_years: int = 5) -> pd.Series:
    """
    Calculate cost of equity using CAPM model:
    Cost of Equity = Risk-free rate + Beta * (Market return - Risk-free rate)
    
    Assumptions:
    - USD Risk-free rate: 4.5% (approximate current US 10-year Treasury)
    - INR Risk-free rate: 7.0% (approximate current Indian 10-year G-Sec)
    - USD Market return: 10% (historical S&P 500 average)
    - INR Market return: 12% (historical NIFTY 50 average)
    """
    
    # Risk-free rates and market returns (annualized)
    USD_RF_RATE = 0.045  # 4.5%
    INR_RF_RATE = 0.070  # 7.0%
    USD_MARKET_RETURN = 0.10  # 10%
    INR_MARKET_RETURN = 0.12  # 12%
    
    # Get beta values
    beta_series = compute_beta_from_df(combined_df, df_portfolio, lookback_years)
    
    cost_of_equity_dict = {}
    
    # Get equity instruments only
    equity_instruments = df_portfolio[
        df_portfolio["Instrument Type"].str.lower().isin(["equity", "stock", "etf"])
    ]
    
    for _, row in equity_instruments.iterrows():
        instrument = row["Name"]
        currency = row["Currency"]
        
        if instrument not in beta_series.index or pd.isna(beta_series[instrument]):
            cost_of_equity_dict[instrument] = np.nan
            continue
        
        beta = beta_series[instrument]
        
        # Choose parameters based on currency
        if currency == "USD":
            risk_free_rate = USD_RF_RATE
            market_return = USD_MARKET_RETURN
        else:  # INR or other
            risk_free_rate = INR_RF_RATE
            market_return = INR_MARKET_RETURN
        
        # CAPM formula
        cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)
        cost_of_equity_dict[instrument] = cost_of_equity
    
    # Add Cash instruments with cost = risk-free rate
    cash_instruments = df_portfolio[
        df_portfolio['Instrument Type'].str.lower().isin(['cash', 'cd', 'post office'])
    ]
    
    for _, row in cash_instruments.iterrows():
        instrument = row["Name"]
        currency = row["Currency"]
        
        if instrument not in cost_of_equity_dict:
            # Cash cost of capital = risk-free rate
            if currency == "USD":
                cost_of_equity_dict[instrument] = USD_RF_RATE
            else:  # INR or other
                cost_of_equity_dict[instrument] = INR_RF_RATE
    
    return pd.Series(cost_of_equity_dict, name="Cost of Equity")


# ─── 8) GET SECTOR ALLOCATION ─────────────────────────────────────────────────

def get_sector_allocation(df_portfolio: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get sector allocation for equity instruments using yfinance info.
    Returns tuple of (instrument_sector_df, sector_allocation_df)
    """
    
    # Get equity instruments only
    equity_instruments = df_portfolio[
        df_portfolio["Instrument Type"].str.lower().isin(["equity", "stock", "etf"])
    ].copy()
    
    if equity_instruments.empty:
        empty_df1 = pd.DataFrame(columns=['Name', 'Sector', 'Current Value INR', 'Weight (%)'])
        empty_df2 = pd.DataFrame(columns=['Sector', 'Total Value INR', 'Weight (%)'])
        return empty_df1, empty_df2
    
    # Get sector information
    sector_dict = {}
    for instrument in equity_instruments["Name"].unique():
        try:
            ticker = yf.Ticker(instrument)
            info = ticker.info
            sector = info.get('sector', 'Unknown')
            if sector is None or sector == '':
                sector = 'Unknown'
            sector_dict[instrument] = sector
        except Exception as e:
            print(f"Error getting sector for {instrument}: {e}")
            sector_dict[instrument] = 'Unknown'
    
    # Add sector information to dataframe
    equity_instruments['Sector'] = equity_instruments['Name'].map(sector_dict)
    
    # Calculate total equity value for percentage calculation
    total_equity_value = equity_instruments['Current Value INR'].sum()
    
    if total_equity_value == 0:
        empty_df1 = pd.DataFrame(columns=['Name', 'Sector', 'Current Value INR', 'Weight (%)'])
        empty_df2 = pd.DataFrame(columns=['Sector', 'Total Value INR', 'Weight (%)'])
        return empty_df1, empty_df2
    
    # Individual instrument sector allocation
    instrument_sector_df = equity_instruments[['Name', 'Sector', 'Current Value INR']].copy()
    instrument_sector_df['Weight (%)'] = (
        instrument_sector_df['Current Value INR'] / total_equity_value * 100
    ).round(2)
    
    # Aggregated sector allocation
    sector_allocation = (
        equity_instruments
        .groupby('Sector')['Current Value INR']
        .sum()
        .reset_index(name='Total Value INR')
    )
    sector_allocation['Weight (%)'] = (
        sector_allocation['Total Value INR'] / total_equity_value * 100
    ).round(2)
    sector_allocation = sector_allocation.sort_values('Total Value INR', ascending=False)
    
    return instrument_sector_df, sector_allocation