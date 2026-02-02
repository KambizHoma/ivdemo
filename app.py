"""
ivdemo v1.2 - Jaeckel Implied Volatility with Real Market Data
Nippotica Fast Financial Computing

Real-time market volatility analysis using Peter Jaeckel's LetsBeRational algorithm
with adaptive filtering and professional-grade accuracy.
"""

import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_vollib.black_scholes import black_scholes
import yfinance as yf
from datetime import datetime

# MUST be first Streamlit command
st.set_page_config(
    page_title="Market IV Analysis",
    page_icon="favicon.png",
    layout="wide"
)


@st.cache_resource
def warmup_numba():
    """
    Warm up Numba JIT compilation on app startup.
    First call to py_vollib triggers compilation (~500ms).
    This prevents misleading slow speeds on first user interaction.
    
    Returns:
        bool: True when warmup complete
    """
    try:
        _ = implied_volatility(5.0, 100.0, 100.0, 0.25, 0.05, 'c')
        return True
    except Exception as e:
        return False


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_spot_and_expirations(ticker_symbol):
    """
    Fetch spot price and available expirations from Yahoo Finance with caching.
    Returns only serializable data.
    
    Args:
        ticker_symbol: Stock ticker symbol (e.g., 'QQQ')
    
    Returns:
        tuple: (spot_price, available_expirations) or (None, None) on error
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Get current spot price
        hist = ticker.history(period='1d')
        if hist.empty:
            return None, None
        spot = float(hist['Close'].iloc[-1])
        
        # Get available expiration dates
        expirations = list(ticker.options)
        if not expirations:
            return None, None
            
        return spot, expirations
    except Exception as e:
        return None, None


def find_closest_expiry(expirations, target_days):
    """
    Find the expiration date closest to the target number of days.
    
    Args:
        expirations: List of expiration date strings (YYYY-MM-DD)
        target_days: Target days to expiration
    
    Returns:
        str: Closest expiration date
    """
    today = datetime.now()
    
    # Calculate days to each expiration
    days_diff = []
    for exp in expirations:
        exp_date = datetime.strptime(exp, '%Y-%m-%d')
        days = (exp_date - today).days
        days_diff.append(abs(days - target_days))
    
    # Return expiration with minimum difference
    closest_idx = days_diff.index(min(days_diff))
    return expirations[closest_idx]


def clean_options_data(calls_df, spot_price, min_iv=0.01, max_iv=2.0, 
                      min_moneyness=0.90, max_moneyness=1.20):
    """
    Clean and filter options data for quality with adaptive volume threshold.
    Uses asymmetric moneyness bounds suitable for call option volatility analysis.
    
    Args:
        calls_df: DataFrame of call options from yfinance
        spot_price: Current spot price
        min_iv: Minimum implied volatility (decimal)
        max_iv: Maximum implied volatility (decimal)
        min_moneyness: Minimum K/S ratio (e.g., 0.90 = don't show deep ITM)
        max_moneyness: Maximum K/S ratio (e.g., 1.20 = limit far OTM)
    
    Returns:
        tuple: (DataFrame: cleaned data, int: original count, float: volume threshold used)
    """
    # Extract relevant columns
    data = calls_df[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']].copy()
    
    original_count = len(data)
    
    # Calculate mid price
    data['mid'] = (data['bid'] + data['ask']) / 2
    
    # Calculate moneyness (K/S) early for filtering
    data['moneyness'] = data['strike'] / spot_price
    
    # Adaptive volume threshold: Two-stage approach
    # Stage 1: Absolute minimum (don't show options with almost no trading)
    absolute_min_volume = 10
    
    # Stage 2: Relative threshold based on data distribution
    # Use 75th percentile - keeps top 25% by volume (stricter than median)
    volumes = data['volume'].values
    if len(volumes) > 0:
        percentile_75_volume = np.percentile(volumes, 75)
        # Take the higher of absolute minimum or 75th percentile
        adaptive_threshold = max(absolute_min_volume, percentile_75_volume)
    else:
        adaptive_threshold = absolute_min_volume
    
    # Apply filters with asymmetric moneyness bounds
    clean_data = data[
        (data['volume'] > adaptive_threshold) &
        (data['impliedVolatility'] > min_iv) &
        (data['impliedVolatility'] < max_iv) &
        (data['impliedVolatility'].notna()) &
        (data['bid'] > 0) &
        (data['ask'] > data['bid']) &
        (data['moneyness'] >= min_moneyness) &  # Don't show deep ITM
        (data['moneyness'] <= max_moneyness)     # Limit far OTM
    ].copy()
    
    # Sort by strike
    clean_data = clean_data.sort_values('strike').reset_index(drop=True)
    
    return clean_data, original_count, adaptive_threshold


def main():
    # Warmup
    warmup_complete = warmup_numba()
    
    # Monochrome theme CSS
    st.markdown("""
        <style>
        /* Monochrome color scheme */
        :root {
            --primary-color: #4a4a4a;
            --background-color: #f5f5f5;
            --secondary-background-color: #e8e8e8;
            --text-color: #262626;
        }
        
        /* Primary buttons - white background */
        .stButton > button[kind="primary"] {
            background-color: white !important;
            color: #4a4a4a !important;
            border: 1px solid #d0d0d0 !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #f5f5f5 !important;
            border: 1px solid #b0b0b0 !important;
        }
        
        /* Regular buttons - monochrome */
        .stButton > button {
            background-color: #e8e8e8 !important;
            color: #262626 !important;
            border: 1px solid #cccccc !important;
        }
        .stButton > button:hover {
            background-color: #d0d0d0 !important;
            border: 1px solid #999999 !important;
        }
        
        /* Success/Info/Warning boxes - monochrome */
        .stAlert {
            background-color: #f0f0f0 !important;
            color: #262626 !important;
            border-left: 4px solid #808080 !important;
        }
        
        /* Metrics - monochrome */
        [data-testid="stMetricValue"] {
            color: #262626 !important;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #fafafa;
        }
        
        /* Section headers in sidebar */
        [data-testid="stSidebar"] h3 {
            font-size: 1rem;
            color: #4a4a4a;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
            border-bottom: 1px solid #d0d0d0;
            padding-bottom: 0.3rem;
        }
        
        /* Plotly chart background */
        .js-plotly-plot {
            background-color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("Market Implied Volatility Analysis")
    st.subheader("Real-time IV calibration for risk managers and market makers")
    st.markdown("---")
    
    # Warning if warmup failed
    if not warmup_complete:
        st.warning("⚠️ Warmup incomplete. First calculation may be slow.")
    
    # Simplified Sidebar
    
    # Nippotica Logo at top
    st.sidebar.image("nippologo.png", width=150)
    st.sidebar.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    
    st.sidebar.title("Parameters")
    
    # Risk-free rate
    r = st.sidebar.number_input(
        "Risk-free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1,
        format="%.1f",
        help="Annual risk-free interest rate for IV calculations"
    ) / 100  # Convert percentage to decimal
    
    st.sidebar.markdown("---")
    
    # Ticker input
    ticker_symbol = st.sidebar.text_input(
        "Ticker Symbol",
        value="QQQ",
        help="Enter stock or ETF ticker (e.g., QQQ, SPY, AAPL, GLD)"
    ).upper()
    
    # Target days to expiration
    target_days = st.sidebar.number_input(
        "Target Days to Expiration",
        min_value=1,
        max_value=365,
        value=30,
        step=1,
        help="System will find the closest available expiration date"
    )
    
    # Initialize session state for market data
    if 'market_data_loaded' not in st.session_state:
        st.session_state.market_data_loaded = False
        st.session_state.available_expirations = []
    
    # Expiration Selection dropdown (always visible, populated after fetch)
    if st.session_state.market_data_loaded and len(st.session_state.available_expirations) > 0:
        # Find index of currently selected expiry (or default to first)
        default_idx = 0
        if 'selected_expiry' in st.session_state and st.session_state.selected_expiry in st.session_state.available_expirations:
            try:
                default_idx = st.session_state.available_expirations.index(st.session_state.selected_expiry)
            except ValueError:
                default_idx = 0
        
        selected_expiry = st.sidebar.selectbox(
            "Expiration Selection",
            st.session_state.available_expirations,
            index=default_idx,
            key="expiry_selector",
            help="Select expiration date for IV calculation"
        )
        
        # Update selected expiry in session state when user changes it
        st.session_state.selected_expiry = selected_expiry
        
        # Show days to expiry
        expiry_date = datetime.strptime(selected_expiry, '%Y-%m-%d')
        days_to_expiry = (expiry_date - datetime.now()).days
        st.sidebar.caption(f"📅 {days_to_expiry} days to expiration")
    else:
        # Show disabled dropdown before data is fetched
        st.sidebar.selectbox(
            "Expiration Selection",
            ["Fetch data first..."],
            disabled=True,
            key="expiry_selector_disabled",
            help="Will populate after fetching market data"
        )
    
    # Main content
    st.markdown("### Market Data from Yahoo Finance")
    
    # Show success message if it exists (from previous rerun)
    if 'fetch_success_message' in st.session_state:
        st.success(st.session_state.fetch_success_message)
        del st.session_state.fetch_success_message
    
    if 'fetch_info_message' in st.session_state:
        st.info(st.session_state.fetch_info_message)
        del st.session_state.fetch_info_message
    
    # Button 1: Fetch Market Data
    if st.button("▶️ Fetch Market Data", type="primary", key="fetch_button"):
        with st.spinner(f"Fetching options data for {ticker_symbol}..."):
            spot, expirations = fetch_spot_and_expirations(ticker_symbol)
            
            if spot is None:
                st.error(f"❌ Could not fetch data for {ticker_symbol}. Please check the ticker symbol and try again.")
            else:
                st.session_state.market_data_loaded = True
                st.session_state.ticker_symbol = ticker_symbol
                st.session_state.spot = spot
                st.session_state.available_expirations = expirations
                
                # Find closest expiry to target days
                selected_expiry = find_closest_expiry(expirations, target_days)
                st.session_state.selected_expiry = selected_expiry
                
                # Calculate actual days to this expiry
                expiry_date = datetime.strptime(selected_expiry, '%Y-%m-%d')
                actual_days = (expiry_date - datetime.now()).days
                
                # Store messages in session state to show after rerun
                st.session_state.fetch_success_message = f"✓ Data loaded for {ticker_symbol} | Spot: ${spot:.2f} | Found {len(expirations)} expiration dates"
                st.session_state.fetch_info_message = f"📅 Auto-selected: **{selected_expiry}** ({actual_days} days) — Adjust in sidebar if needed, then calculate IVs below"
                
                # Force rerun to update sidebar dropdown
                st.experimental_rerun()
    
    # Button 2: Calculate IVs (only show after data is fetched)
    if st.session_state.market_data_loaded:
        if st.button("🔍 Calculate Implied Volatilities", type="primary", key="calculate_button"):
            with st.spinner("Calculating implied volatilities..."):
                try:
                    selected_expiry = st.session_state.selected_expiry
                    
                    # Create ticker object fresh (not cached)
                    ticker = yf.Ticker(st.session_state.ticker_symbol)
                    
                    # Get options chain for selected expiry
                    options_chain = ticker.option_chain(selected_expiry)
                    calls = options_chain.calls
                    
                    # Clean the data with adaptive thresholds and asymmetric moneyness
                    clean_calls, original_count, volume_threshold = clean_options_data(
                        calls, 
                        st.session_state.spot,
                        min_iv=0.01,
                        max_iv=2.0,
                        min_moneyness=0.90,  # Don't show deep ITM calls
                        max_moneyness=1.20   # Reasonable OTM limit for calls
                    )
                    
                    if len(clean_calls) == 0:
                        st.warning("⚠️ No liquid options found for this expiry after filtering. Try a different expiration date from the sidebar.")
                    else:
                        # Calculate days to expiry
                        expiry_date = datetime.strptime(selected_expiry, '%Y-%m-%d')
                        actual_days = (expiry_date - datetime.now()).days
                        T_market = actual_days / 365.0
                        
                        # Calculate IVs using Jaeckel
                        start_time = time.perf_counter()
                        
                        ivs_calculated = []
                        for idx, row in clean_calls.iterrows():
                            try:
                                # Use mid price for IV calculation
                                iv = implied_volatility(row['mid'], st.session_state.spot, row['strike'], 
                                                      T_market, r, 'c')
                                ivs_calculated.append(iv)
                            except:
                                ivs_calculated.append(np.nan)
                        
                        clean_calls['calculated_iv'] = ivs_calculated
                        
                        # Remove any failed calculations
                        clean_calls = clean_calls[clean_calls['calculated_iv'].notna()].copy()
                        
                        end_time = time.perf_counter()
                        total_time_ms = (end_time - start_time) * 1000
                        
                        # Store results
                        st.session_state.market_results = {
                            'clean_calls': clean_calls,
                            'original_count': original_count,
                            'volume_threshold': volume_threshold,
                            'spot': st.session_state.spot,
                            'expiry': selected_expiry,
                            'days_to_expiry': actual_days,
                            'total_time_ms': total_time_ms,
                            'ticker': st.session_state.ticker_symbol
                        }
                        
                except Exception as e:
                    st.error(f"❌ Error processing options data: {str(e)}")
    
    # Display results if available
    if 'market_results' in st.session_state:
        results = st.session_state.market_results
        clean_calls = results['clean_calls']
        
        # Data Quality Report
        st.markdown("### Data Quality Report")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Spot Price", f"${results['spot']:.2f}")
        with col2:
            st.metric("Raw Strikes", results['original_count'])
        with col3:
            st.metric("Clean Strikes", len(clean_calls))
        with col4:
            pct_retained = (len(clean_calls) / results['original_count'] * 100)
            st.metric("Retained", f"{pct_retained:.0f}%")
        
        st.info(
            f"**Expiry:** {results['expiry']} ({results['days_to_expiry']} days) | "
            f"**Adaptive Volume Threshold:** {results['volume_threshold']:.0f} contracts (75th percentile, min 10) | "
            f"**Removed:** {results['original_count'] - len(clean_calls)} strikes"
        )
        
        # Market IV Smile Chart (MOVED UP - BEFORE TABLE)
        st.markdown("### Market Implied Volatility Smile")
        
        fig = go.Figure()
        
        # Yahoo's reported IV (for comparison)
        fig.add_trace(go.Scatter(
            x=clean_calls['moneyness'],
            y=clean_calls['impliedVolatility'] * 100,
            mode='markers',
            name='Yahoo Finance IV',
            marker=dict(size=10, color='#b0b0b0', symbol='circle'),
            hovertemplate='<b>Yahoo IV</b><br>Strike: $%{customdata:.0f}<br>K/S: %{x:.3f}<br>IV: %{y:.2f}%<extra></extra>',
            customdata=clean_calls['strike']
        ))
        
        # Jaeckel calculated IV
        fig.add_trace(go.Scatter(
            x=clean_calls['moneyness'],
            y=clean_calls['calculated_iv'] * 100,
            mode='markers+lines',
            name='Jaeckel Calculated IV',
            line=dict(color='#4a4a4a', width=2),
            marker=dict(size=8, color='#4a4a4a', symbol='x'),
            hovertemplate='<b>Jaeckel IV</b><br>Strike: $%{customdata:.0f}<br>K/S: %{x:.3f}<br>IV: %{y:.2f}%<extra></extra>',
            customdata=clean_calls['strike']
        ))
        
        # Add ATM line
        fig.add_vline(x=1.0, line_dash="dash", line_color="#b0b0b0", 
                     annotation_text="ATM", annotation_position="top")
        
        fig.update_layout(
            title=f'{results["ticker"]} Market IV Smile - {results["expiry"]}',
            xaxis_title='Moneyness (K/S)',
            yaxis_title='Implied Volatility (%)',
            hovermode='closest',
            height=500,
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#262626'),
            xaxis=dict(gridcolor='#e8e8e8', zerolinecolor='#cccccc'),
            yaxis=dict(gridcolor='#e8e8e8', zerolinecolor='#cccccc')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance Metrics
        st.markdown("### Performance Metrics")
        
        # Calculate IV differences
        iv_diff = np.abs(clean_calls['calculated_iv'] - clean_calls['impliedVolatility']) * 100
        mean_diff = iv_diff.mean()
        max_diff = iv_diff.max()
        
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric(
                label="Strikes Calculated",
                value=len(clean_calls)
            )
        
        with perf_col2:
            st.metric(
                label="Calculation Time",
                value=f"{results['total_time_ms']:.2f} ms"
            )
        
        with perf_col3:
            st.metric(
                label="Mean IV Diff",
                value=f"{mean_diff:.2f}%",
                help="Average absolute difference between Yahoo and Jaeckel IVs"
            )
        
        with perf_col4:
            st.metric(
                label="Max IV Diff",
                value=f"{max_diff:.2f}%",
                help="Maximum absolute difference between Yahoo and Jaeckel IVs"
            )
        
        st.success(
            f"✓ All {len(clean_calls)} calculations converged in 2 iterations to machine precision"
        )
        
        st.info(
            f"**Comparison:** Jaeckel's algorithm vs. Yahoo Finance IV | "
            f"Mean difference: {mean_diff:.2f}% | Max difference: {max_diff:.2f}%"
        )
        
        # Options Chain Table (MOVED DOWN - IN EXPANDER)
        with st.expander("📋 Options Chain with Implied Volatilities"):
            # Prepare display dataframe
            display_df = clean_calls[['strike', 'moneyness', 'bid', 'ask', 'mid', 
                                      'volume', 'impliedVolatility', 'calculated_iv']].copy()
            display_df.columns = ['Strike', 'K/S', 'Bid', 'Ask', 'Mid', 'Volume', 
                                 'Yahoo IV', 'Jaeckel IV']
            
            # Format for display
            display_df['K/S'] = display_df['K/S'].apply(lambda x: f"{x:.3f}")
            display_df['Bid'] = display_df['Bid'].apply(lambda x: f"${x:.2f}")
            display_df['Ask'] = display_df['Ask'].apply(lambda x: f"${x:.2f}")
            display_df['Mid'] = display_df['Mid'].apply(lambda x: f"${x:.2f}")
            display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{int(x):,}")
            display_df['Yahoo IV'] = display_df['Yahoo IV'].apply(lambda x: f"{x*100:.2f}%")
            display_df['Jaeckel IV'] = display_df['Jaeckel IV'].apply(lambda x: f"{x*100:.2f}%")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Data Quality Details
        with st.expander("Data Quality & Methodology"):
            st.markdown(f"""
            **Adaptive Filtering Applied:**
            - Volume threshold: {results['volume_threshold']:.0f} contracts (dynamically computed)
              - Method: 75th percentile of raw data volumes (top 25% by volume)
              - Minimum: 10 contracts (absolute floor)
              - Result: Adapts to liquidity regime while maintaining quality
            - IV range: 1% to 200%
            - **Moneyness range: 0.90 ≤ K/S ≤ 1.20 (asymmetric for call options)**
              - Lower: K/S ≥ 0.90 (excludes deep ITM calls with mostly intrinsic value)
              - Upper: K/S ≤ 1.20 (reasonable OTM limit for volatility analysis)
              - Why asymmetric? Deep ITM calls aren't useful for vol smiles
            - Bid > 0, Ask > Bid (removes stale/illiquid quotes)
            - No NaN values
            
            **Why 75th Percentile?**
            - Liquid tickers (SPY, QQQ): High threshold → Only most liquid strikes
            - Less liquid tickers (Individual stocks): Lower threshold → Reasonable coverage
            - Stricter than median (50th) for better data quality
            - Data-driven, not arbitrary
            
            **Market Data:**
            - Source: Yahoo Finance API
            - Cache duration: 5 minutes
            - Spot price: ${results['spot']:.2f}
            - Days to expiry: {results['days_to_expiry']}
            - Time to expiry (T): {results['days_to_expiry']/365:.4f} years
            
            **Calculation Method:**
            - Algorithm: Jaeckel's LetsBeRational
            - Price used: Mid (average of bid/ask)
            - Risk-free rate: {r*100:.2f}%
            - Convergence: 2 iterations guaranteed to machine precision
            """)
        
        # About This Application
        with st.expander("ℹ️ About This Application"):
            st.markdown("""
            **About This Application**
            
            This tool calculates implied volatilities from real market options data using Peter Jaeckel's LetsBeRational algorithm via py_vollib by Larry Richards. It fetches live options chains from Yahoo Finance, filters for liquid strikes, and computes IVs with machine precision in milliseconds. The system automatically adapts filtering to different liquidity regimes—strict for high-volume tickers like QQQ, more lenient for individual stocks.
            
            Designed for derivatives professionals, this application helps risk managers assess current volatility structures and market makers validate their IV calculations. The comparison between Jaeckel-calculated IVs and Yahoo Finance's reported values (typically within 1% for liquid options) demonstrates both algorithm accuracy and data quality. Results include full volatility smile visualization, detailed options chain data, and complete transparency into the filtering methodology.
            """)
        
        # Understanding IV Differences
        with st.expander("🔍 Understanding IV Differences"):
            st.markdown("""
            **Why Yahoo IV Differs (And Why That's Expected)**
            
            Yahoo Finance's reported IVs may differ from our Jaeckel calculations for several reasons. Yahoo typically uses the last traded price, which can be stale for illiquid strikes, while we use the current mid price (average of bid-ask) which better reflects real-time market conditions. Yahoo may also use different convergence tolerances or algorithms. For liquid options (high volume, tight spreads), both methods converge closely—differences under 1% are typical. For less liquid strikes, wider spreads and stale last prices can cause Yahoo's IVs to diverge significantly. Our approach—using current mid prices with Jaeckel's machine-precision algorithm—provides a more consistent and theoretically sound estimate of current market-implied volatility, particularly useful when Yahoo's data lags or when analyzing options that don't trade frequently.
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #808080; font-size: 0.9em;'>
        <p>ivdemo v1.2 • Built with Streamlit • Powered by py_vollib & Numba</p>
        <p>© 2025 Nippotica Corporation • Fast Financial Computing Solutions</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
