import yfinance as yf
import pandas as pd
import numpy as np


# =========================
# CONFIGURATION
# =========================
TICKER = "SPY"
START_DATE = "2020-01-01"
END_DATE = "2026-01-01"
SHORT_WINDOW = 20
LONG_WINDOW = 50


# =========================
# DOWNLOAD DATA
# =========================
def download_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)

    if "Close" in data.columns:
        prices = data["Close"].copy()
    else:
        prices = data.copy()

    prices = prices.dropna()
    return prices


# =========================
# BUILD STRATEGY
# =========================
def build_strategy_dataframe(prices, short_window, long_window):
    df = pd.DataFrame()

    df["Close"] = prices #Load the prices to the dataframe

    df["MA_Short"] = df["Close"].rolling(window=short_window).mean()# computes the average on the last 20 days 
    df["MA_Long"] = df["Close"].rolling(window=long_window).mean() #computes the average on the las 50 days

    # Signal: 1 when short MA > long MA, else 0
    df["Signal"] = np.where(df["MA_Short"] > df["MA_Long"], 1, 0)
    
    # Position is yesterday's signal to avoid look-ahead bias
    df["Position"] = df["Signal"].shift(1).fillna(0)

    df["Market Return"] = df["Close"].pct_change() #percentage change between consecutive values in a dataset
    df["Strategy Return"] = df["Position"] * df["Market Return"]

    df = df.dropna() # delate enpty values
    return df


# =========================
# METRICS
# =========================
def cumulative_returns(returns):
    return (1 + returns).cumprod() - 1 


def annualized_volatility(returns):
    return returns.std() * np.sqrt(252)


def max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1
    return drawdown.min()


def count_trades(signal_series):
    # Count changes in signal from 0 to 1 and 1 to 0
    changes = signal_series.diff().fillna(0) #calculates the difference between a value and a previous value.
    trades = (changes != 0).sum() #summarize
    return int(trades)


def calculate_trade_win_rate(df):
    trades = []
    in_trade = False
    entry_price = 0

    for i in range(len(df)):
        current_signal = df["Signal"].iloc[i] #take out "signal"
        current_price = df["Close"].iloc[i] #take out signal

        if not in_trade and current_signal == 1:
            in_trade = True
            entry_price = current_price

        elif in_trade and current_signal == 0:
            in_trade = False
            exit_price = current_price
            trade_return = (exit_price / entry_price) - 1
            trades.append(trade_return) #It adds trade_return to the list trades

    # If strategy is still in a trade at the end, close it at last price
    if in_trade:
        exit_price = df["Close"].iloc[-1]
        trade_return = (exit_price / entry_price) - 1
        trades.append(trade_return)

    if len(trades) == 0:
        return 0, 0, []

    wins = sum(1 for trade in trades if trade > 0)
    win_rate = wins / len(trades)

    return len(trades), win_rate, trades


def build_summary(df):
    market_cum_return = cumulative_returns(df["Market Return"])
    strategy_cum_return = cumulative_returns(df["Strategy Return"])

    total_trades_signal = count_trades(df["Signal"])
    completed_trades, win_rate, trade_list = calculate_trade_win_rate(df)

    summary = {
        "Buy and Hold Total Return": market_cum_return.iloc[-1],
        "Strategy Total Return": strategy_cum_return.iloc[-1],
        "Buy and Hold Volatility": annualized_volatility(df["Market Return"]),
        "Strategy Volatility": annualized_volatility(df["Strategy Return"]),
        "Buy and Hold Max Drawdown": max_drawdown(df["Market Return"]),
        "Strategy Max Drawdown": max_drawdown(df["Strategy Return"]),
        "Signal Changes": total_trades_signal,
        "Completed Trades": completed_trades,
        "Win Rate": win_rate,
    }

    return summary, market_cum_return, strategy_cum_return, trade_list


# =========================
# MAIN
# =========================
def main():
    print(f"Downloading data for {TICKER}...")
    prices = download_data(TICKER, START_DATE, END_DATE)

    df = build_strategy_dataframe(prices, SHORT_WINDOW, LONG_WINDOW)
    summary, market_curve, strategy_curve, trade_list = build_summary(df)

    print("\n================ STRATEGY DATA ================\n")
    print(df.tail())

    print("\n================ BACKTEST SUMMARY ================\n")
    for key, value in summary.items():
        if "Return" in key or "Volatility" in key or "Drawdown" in key or "Rate" in key:
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value}")

    print("\n================ LAST 5 CUMULATIVE RETURNS ================\n")
    comparison = pd.DataFrame({
        "Buy_and_Hold": market_curve,
        "Strategy": strategy_curve
    })
    print(comparison.tail())

    if trade_list:
        print("\n================ TRADE RETURNS ================\n")
        for i, trade_return in enumerate(trade_list, start=1):
            print(f"Trade {i}: {trade_return:.2%}")


if __name__ == "__main__":
    main()