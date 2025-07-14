from typing import Annotated, Dict
from .reddit_utils import fetch_top_from_category
from .yfin_utils import *
from .stockstats_utils import *
from .googlenews_utils import *
from .finnhub_utils import get_data_in_range
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
import pandas as pd
from tqdm import tqdm
import yfinance as yf
from openai import OpenAI
from .config import get_config, set_config, DATA_DIR


def get_finnhub_news(
    ticker: Annotated[
        str,
        "Search query of a company's, e.g. 'AAPL, TSM, etc.",
    ],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
):
    """
    Retrieve news about a company within a time frame

    Args
        ticker (str): ticker for the company you are interested in
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns
        str: dataframe containing the news of the company in the time frame

    """

    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    result = get_data_in_range(ticker, before, curr_date, "news_data", DATA_DIR)

    if len(result) == 0:
        return ""

    combined_result = ""
    for day, data in result.items():
        if len(data) == 0:
            continue
        for entry in data:
            current_news = (
                "### " + entry["headline"] + f" ({day})" + "\n" + entry["summary"]
            )
            combined_result += current_news + "\n\n"

    return f"## {ticker} News, from {before} to {curr_date}:\n" + str(combined_result)


def get_finnhub_company_insider_sentiment(
    ticker: Annotated[str, "ticker symbol for the company"],
    curr_date: Annotated[
        str,
        "current date of you are trading at, yyyy-mm-dd",
    ],
    look_back_days: Annotated[int, "number of days to look back"],
):
    """
    Retrieve insider sentiment about a company (retrieved from public SEC information) for the past 15 days
    Args:
        ticker (str): ticker symbol of the company
        curr_date (str): current date you are trading on, yyyy-mm-dd
    Returns:
        str: a report of the sentiment in the past 15 days starting at curr_date
    """

    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    data = get_data_in_range(ticker, before, curr_date, "insider_senti", DATA_DIR)

    if len(data) == 0:
        return ""

    result_str = ""
    seen_dicts = []
    for date, senti_list in data.items():
        for entry in senti_list:
            if entry not in seen_dicts:
                result_str += f"### {entry['year']}-{entry['month']}:\nChange: {entry['change']}\nMonthly Share Purchase Ratio: {entry['mspr']}\n\n"
                seen_dicts.append(entry)

    return (
        f"## {ticker} Insider Sentiment Data for {before} to {curr_date}:\n"
        + result_str
        + "The change field refers to the net buying/selling from all insiders' transactions. The mspr field refers to monthly share purchase ratio."
    )


def get_finnhub_company_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[
        str,
        "current date you are trading at, yyyy-mm-dd",
    ],
    look_back_days: Annotated[int, "how many days to look back"],
):
    """
    Retrieve insider transcaction information about a company (retrieved from public SEC information) for the past 15 days
    Args:
        ticker (str): ticker symbol of the company
        curr_date (str): current date you are trading at, yyyy-mm-dd
    Returns:
        str: a report of the company's insider transaction/trading informtaion in the past 15 days
    """

    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    data = get_data_in_range(ticker, before, curr_date, "insider_trans", DATA_DIR)

    if len(data) == 0:
        return ""

    result_str = ""

    seen_dicts = []
    for date, senti_list in data.items():
        for entry in senti_list:
            if entry not in seen_dicts:
                result_str += f"### Filing Date: {entry['filingDate']}, {entry['name']}:\nChange:{entry['change']}\nShares: {entry['share']}\nTransaction Price: {entry['transactionPrice']}\nTransaction Code: {entry['transactionCode']}\n\n"
                seen_dicts.append(entry)

    return (
        f"## {ticker} insider transactions from {before} to {curr_date}:\n"
        + result_str
        + "The change field reflects the variation in share count—here a negative number indicates a reduction in holdings—while share specifies the total number of shares involved. The transactionPrice denotes the per-share price at which the trade was executed, and transactionDate marks when the transaction occurred. The name field identifies the insider making the trade, and transactionCode (e.g., S for sale) clarifies the nature of the transaction. FilingDate records when the transaction was officially reported, and the unique id links to the specific SEC filing, as indicated by the source. Additionally, the symbol ties the transaction to a particular company, isDerivative flags whether the trade involves derivative securities, and currency notes the currency context of the transaction."
    )


def get_simfin_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        DATA_DIR,
        "fundamental_data",
        "simfin_data_all",
        "balance_sheet",
        "companies",
        "us",
        f"us-balance-{freq}.csv",
    )
    df = pd.read_csv(data_path, sep=";")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        print("No balance sheet available before the given current date.")
        return ""

    # Get the most recent balance sheet by selecting the row with the latest Publish Date
    latest_balance_sheet = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
    latest_balance_sheet = latest_balance_sheet.drop("SimFinId")

    return (
        f"## {freq} balance sheet for {ticker} released on {str(latest_balance_sheet['Publish Date'])[0:10]}: \n"
        + str(latest_balance_sheet)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a breakdown of assets, liabilities, and equity. Assets are grouped as current (liquid items like cash and receivables) and noncurrent (long-term investments and property). Liabilities are split between short-term obligations and long-term debts, while equity reflects shareholder funds such as paid-in capital and retained earnings. Together, these components ensure that total assets equal the sum of liabilities and equity."
    )


def get_simfin_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        DATA_DIR,
        "fundamental_data",
        "simfin_data_all",
        "cash_flow",
        "companies",
        "us",
        f"us-cashflow-{freq}.csv",
    )
    df = pd.read_csv(data_path, sep=";")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        print("No cash flow statement available before the given current date.")
        return ""

    # Get the most recent cash flow statement by selecting the row with the latest Publish Date
    latest_cash_flow = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
    latest_cash_flow = latest_cash_flow.drop("SimFinId")

    return (
        f"## {freq} cash flow statement for {ticker} released on {str(latest_cash_flow['Publish Date'])[0:10]}: \n"
        + str(latest_cash_flow)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a breakdown of cash movements. Operating activities show cash generated from core business operations, including net income adjustments for non-cash items and working capital changes. Investing activities cover asset acquisitions/disposals and investments. Financing activities include debt transactions, equity issuances/repurchases, and dividend payments. The net change in cash represents the overall increase or decrease in the company's cash position during the reporting period."
    )


def get_simfin_income_statements(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        DATA_DIR,
        "fundamental_data",
        "simfin_data_all",
        "income_statements",
        "companies",
        "us",
        f"us-income-{freq}.csv",
    )
    df = pd.read_csv(data_path, sep=";")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        print("No income statement available before the given current date.")
        return ""

    # Get the most recent income statement by selecting the row with the latest Publish Date
    latest_income = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
    latest_income = latest_income.drop("SimFinId")

    return (
        f"## {freq} income statement for {ticker} released on {str(latest_income['Publish Date'])[0:10]}: \n"
        + str(latest_income)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a comprehensive breakdown of the company's financial performance. Starting with Revenue, it shows Cost of Revenue and resulting Gross Profit. Operating Expenses are detailed, including SG&A, R&D, and Depreciation. The statement then shows Operating Income, followed by non-operating items and Interest Expense, leading to Pretax Income. After accounting for Income Tax and any Extraordinary items, it concludes with Net Income, representing the company's bottom-line profit or loss for the period."
    )


def get_google_news(
    query: Annotated[str, "Query to search with"],
    curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    query = query.replace(" ", "+")

    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    news_results = getNewsData(query, before, curr_date)

    news_str = ""

    for news in news_results:
        news_str += (
            f"### {news['title']} (source: {news['source']}) \n\n{news['snippet']}\n\n"
        )

    if len(news_results) == 0:
        return ""

    return f"## {query} Google News, from {before} to {curr_date}:\n\n{news_str}"


def get_reddit_global_news(
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
    max_limit_per_day: Annotated[int, "Maximum number of news per day"],
) -> str:
    """
    Retrieve the latest top reddit news
    Args:
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the latest news articles posts on reddit and meta information in these columns: "created_utc", "id", "title", "selftext", "score", "num_comments", "url"
    """

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    posts = []
    # iterate from start_date to end_date
    curr_date = datetime.strptime(before, "%Y-%m-%d")

    total_iterations = (start_date - curr_date).days + 1
    pbar = tqdm(desc=f"Getting Global News on {start_date}", total=total_iterations)

    while curr_date <= start_date:
        curr_date_str = curr_date.strftime("%Y-%m-%d")
        fetch_result = fetch_top_from_category(
            "global_news",
            curr_date_str,
            max_limit_per_day,
            data_path=os.path.join(DATA_DIR, "reddit_data"),
        )
        posts.extend(fetch_result)
        curr_date += relativedelta(days=1)
        pbar.update(1)

    pbar.close()

    if len(posts) == 0:
        return ""

    news_str = ""
    for post in posts:
        if post["content"] == "":
            news_str += f"### {post['title']}\n\n"
        else:
            news_str += f"### {post['title']}\n\n{post['content']}\n\n"

    return f"## Global News Reddit, from {before} to {curr_date}:\n{news_str}"


def get_reddit_company_news(
    ticker: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
    max_limit_per_day: Annotated[int, "Maximum number of news per day"],
) -> str:
    """
    Retrieve the latest top reddit news
    Args:
        ticker: ticker symbol of the company
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the latest news articles posts on reddit and meta information in these columns: "created_utc", "id", "title", "selftext", "score", "num_comments", "url"
    """

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    posts = []
    # iterate from start_date to end_date
    curr_date = datetime.strptime(before, "%Y-%m-%d")

    total_iterations = (start_date - curr_date).days + 1
    pbar = tqdm(
        desc=f"Getting Company News for {ticker} on {start_date}",
        total=total_iterations,
    )

    while curr_date <= start_date:
        curr_date_str = curr_date.strftime("%Y-%m-%d")
        fetch_result = fetch_top_from_category(
            "company_news",
            curr_date_str,
            max_limit_per_day,
            ticker,
            data_path=os.path.join(DATA_DIR, "reddit_data"),
        )
        posts.extend(fetch_result)
        curr_date += relativedelta(days=1)

        pbar.update(1)

    pbar.close()

    if len(posts) == 0:
        return ""

    news_str = ""
    for post in posts:
        if post["content"] == "":
            news_str += f"### {post['title']}\n\n"
        else:
            news_str += f"### {post['title']}\n\n{post['content']}\n\n"

    return f"##{ticker} News Reddit, from {before} to {curr_date}:\n\n{news_str}"


def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd. If no data for the date, will use the latest available data."
    ],
    look_back_days: Annotated[int, "how many days to look back"],
    online: Annotated[bool, "to fetch data online or offline"],
) -> str:

    best_ind_params = {
        # Moving Averages
        "close_50_sma": (
            "50 SMA: A medium-term trend indicator. "
            "Usage: Identify trend direction and serve as dynamic support/resistance. "
            "Tips: It lags price; combine with faster indicators for timely signals."
        ),
        "close_200_sma": (
            "200 SMA: A long-term trend benchmark. "
            "Usage: Confirm overall market trend and identify golden/death cross setups. "
            "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
        ),
        "close_10_ema": (
            "10 EMA: A responsive short-term average. "
            "Usage: Capture quick shifts in momentum and potential entry points. "
            "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
        ),
        # MACD Related
        "macd": (
            "MACD: Computes momentum via differences of EMAs. "
            "Usage: Look for crossovers and divergence as signals of trend changes. "
            "Tips: Confirm with other indicators in low-volatility or sideways markets."
        ),
        "macds": (
            "MACD Signal: An EMA smoothing of the MACD line. "
            "Usage: Use crossovers with the MACD line to trigger trades. "
            "Tips: Should be part of a broader strategy to avoid false positives."
        ),
        "macdh": (
            "MACD Histogram: Shows the gap between the MACD line and its signal. "
            "Usage: Visualize momentum strength and spot divergence early. "
            "Tips: Can be volatile; complement with additional filters in fast-moving markets."
        ),
        # Momentum Indicators
        "rsi": (
            "RSI: Measures momentum to flag overbought/oversold conditions. "
            "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
            "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
        ),
        # Volatility Indicators
        "boll": (
            "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
            "Usage: Acts as a dynamic benchmark for price movement. "
            "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
        ),
        "boll_ub": (
            "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
            "Usage: Signals potential overbought conditions and breakout zones. "
            "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
        ),
        "boll_lb": (
            "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
            "Usage: Indicates potential oversold conditions. "
            "Tips: Use additional analysis to avoid false reversal signals."
        ),
        "atr": (
            "ATR: Averages true range to measure volatility. "
            "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
            "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
        ),
        # Volume-Based Indicators
        "vwma": (
            "VWMA: A moving average weighted by volume. "
            "Usage: Confirm trends by integrating price action with volume data. "
            "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
        ),
        "mfi": (
            "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
            "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
            "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
        ),
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(best_ind_params.keys())}"
        )

    end_date = curr_date
    curr_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date - relativedelta(days=look_back_days)

    if not online:
        # read from YFin data
        data = pd.read_csv(
            os.path.join(
                DATA_DIR,
                f"market_data/price_data/{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
            )
        )
        data["Date"] = pd.to_datetime(data["Date"], utc=True)
        dates_in_df = data["Date"].astype(str).str[:10]

        ind_string = ""
        while curr_date >= before:
            # only do the trading dates
            if curr_date.strftime("%Y-%m-%d") in dates_in_df.values:
                indicator_value = get_stockstats_indicator(
                    symbol, indicator, curr_date.strftime("%Y-%m-%d"), online
                )

                ind_string += f"{curr_date.strftime('%Y-%m-%d')}: {indicator_value}\n"

            curr_date = curr_date - relativedelta(days=1)
    else:
        # online gathering
        ind_string = ""
        while curr_date >= before:
            indicator_value = get_stockstats_indicator(
                symbol, indicator, curr_date.strftime("%Y-%m-%d"), online
            )

            ind_string += f"{curr_date.strftime('%Y-%m-%d')}: {indicator_value}\n"

            curr_date = curr_date - relativedelta(days=1)

    result_str = (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
    )

    return result_str


def get_stockstats_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd. If no data for the date, will use the latest available data."
    ],
    online: Annotated[bool, "to fetch data online or offline"],
) -> str:

    curr_date = datetime.strptime(curr_date, "%Y-%m-%d")
    curr_date = curr_date.strftime("%Y-%m-%d")

    try:
        indicator_value = StockstatsUtils.get_stock_stats(
            symbol,
            indicator,
            curr_date,
            os.path.join(DATA_DIR, "market_data", "price_data"),
            online=online,
        )
    except Exception as e:
        print(
            f"Error getting stockstats indicator data for indicator {indicator} on {curr_date}: {e}"
        )
        return ""

    return str(indicator_value)


def get_YFin_data_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    # calculate past days
    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    start_date = before.strftime("%Y-%m-%d")

    # read in data
    data = pd.read_csv(
        os.path.join(
            DATA_DIR,
            f"market_data/price_data/{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
        )
    )

    # Extract just the date part for comparison
    data["DateOnly"] = data["Date"].str[:10]

    # Filter data between the start and end dates (inclusive)
    filtered_data = data[
        (data["DateOnly"] >= start_date) & (data["DateOnly"] <= curr_date)
    ]

    # Drop the temporary column we created
    filtered_data = filtered_data.drop("DateOnly", axis=1)

    # Set pandas display options to show the full DataFrame
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", None
    ):
        df_string = filtered_data.to_string()

    return (
        f"## Raw Market Data for {symbol} from {start_date} to {curr_date}:\n\n"
        + df_string
    )


def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format, inclusive"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format, exclusive (data returned up to but not including this date)"],
):

    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    # Create ticker object
    ticker = yf.Ticker(symbol.upper())

    # Fetch historical data for the specified date range
    data = ticker.history(start=start_date, end=end_date)

    # Check if data is empty
    if data.empty:
        return (
            f"No data found for symbol '{symbol}' between {start_date} and {end_date}"
        )

    # Remove timezone info from index for cleaner output
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # Round numerical values to 2 decimal places for cleaner display
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].round(2)

    # Convert DataFrame to CSV string
    csv_string = data.to_csv()

    # Add header information
    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string


def get_YFin_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    # read in data
    data = pd.read_csv(
        os.path.join(
            DATA_DIR,
            f"market_data/price_data/{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
        )
    )

    if end_date > "2025-03-25":
        raise Exception(
            f"Get_YFin_Data: {end_date} is outside of the data range of 2015-01-01 to 2025-03-25"
        )

    # Extract just the date part for comparison
    data["DateOnly"] = data["Date"].str[:10]

    # Filter data between the start and end dates (inclusive)
    filtered_data = data[
        (data["DateOnly"] >= start_date) & (data["DateOnly"] <= end_date)
    ]

    # Drop the temporary column we created
    filtered_data = filtered_data.drop("DateOnly", axis=1)

    # remove the index from the dataframe
    filtered_data = filtered_data.reset_index(drop=True)

    return filtered_data


def get_social_sentiment_openai(ticker, curr_date):
    config = get_config()
    client = OpenAI(base_url=config["backend_url"])

    from datetime import datetime, timedelta
    if isinstance(curr_date, str):
        curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    else:
        curr_date_dt = curr_date
    start_date = (curr_date_dt - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = curr_date_dt.strftime("%Y-%m-%d")

    prompt = f"""
You are a social media narrative analyst bot. Your sole task is to use the web search tool to identify and summarize the dominant public narratives about {ticker} from {start_date} to {end_date}.

CRITICAL INSTRUCTIONS:
1.  Search on sources like Twitter/X, Reddit, StockTwits, etc.
2.  Your primary goal is to identify the **main reasons** people are bullish or bearish.
3.  For each distinct narrative (e.g., "excitement about AI chip," "concern over competition"), create a single entry in the table.
4.  Provide a rough estimate of how prevalent each narrative is (e.g., High, Medium, Low).
5.  Do NOT invent quantitative scores. Do NOT include official company news. Focus ONLY on public discourse.

The output format must be this exact markdown table:

| Type | Dominant Narrative/Theme | Prevalence | Representative Post Summary |
|---|---|---|---|
| Bullish | [The most common positive theme] | [e.g., High] | [A brief summary of a popular post supporting this theme] |
| Bullish | [A second distinct positive theme] | [e.g., Medium] | [A brief summary of a popular post supporting this theme] |
| Bearish | [The most common negative theme] | [e.g., High] | [A brief summary of a popular post supporting this theme] |
"""

# """You are a professional financial news and social sentiment analyst.
# Please search for recent posts and discussions related to the stock company with ticker symbol {ticker}, published from 
# {start_date} up to {end_date}, only from reputable social media and financial community sources such as Twitter, 
# StockTwits, Reddit, Seeking Alpha, and Yahoo Finance discussion boards.
# Focus on posts that discuss company news, earnings, rumors, analyst opinions, and significant market sentiment shifts.
# Present the results in a markdown table with columns: Date, Platform, Author (if available), Headline or Main Point, 
# Summary, and Link.
# Only include posts published within the specified period. Rank the posts by potential trading impact, and cite the 
# source for each entry."""

    response = client.responses.create(
        #todo only for American stocks now
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        #"text": f"Can you search Social Media for {ticker} from 7 days before {curr_date} to {curr_date}? Make sure you only get the data posted during that period.",
                        "text": prompt,
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[
            {
                "type": "web_search_preview",
                # "user_location": {"type": "approximate"},
                "search_context_size": "medium",  # low/medium/high
                # 更精准可控 "time_range": {"start": "2024-06-01", "end": "2024-06-07"}
                # language
                # max_results
                # region
                # custom_query 自定义关键词
            }
        ],
        temperature=0.7,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    return response.output[1].content[0].text


def get_company_news_openai(ticker, curr_date):
    config = get_config()
    client = OpenAI(base_url=config["backend_url"])

    from datetime import datetime, timedelta

    if isinstance(curr_date, str):
        curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    else:
        curr_date_dt = curr_date
    start_date = (curr_date_dt - timedelta(days=14)).strftime("%Y-%m-%d")
    end_date = curr_date_dt.strftime("%Y-%m-%d")

    prompt = f"""
You are an automated news extraction bot. Your sole task is to use the web search tool to find and categorize company-specific news for {ticker}, published from {start_date} up to {end_date}.

CRITICAL INSTRUCTIONS:
1.  Search for factual news ONLY within these specific categories:
    - Earnings & Financials (Results, Guidance, Analyst Ratings)
    - Products & Services (Launches, R&D News)
    - Management & Operations (Executive changes, Layoffs)
    - Legal, Regulatory & M&A
2.  Do NOT include social media discussions, rumors, or sentiment analysis. Focus on official news from reputable financial media.
3.  For each piece of news found, present it as a single line in a markdown table.
4.  Do NOT add any commentary, analysis, ranking, or opinions. Report the facts only.
5.  If no significant news is found for a category, omit it from the table.

The output format must be this exact markdown table:

| Category | Headline | Summary |
|---|---|---|
| [e.g., Earnings] | [Headline of the news] | [A brief, factual summary of the news content] |
| [e.g., Products] | [Headline of the news] | [A brief, factual summary of the news content] |
"""

    response = client.responses.create(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[{"type": "web_search_preview", "search_context_size": "medium"}],
        temperature=0.7,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    return response.output[1].content[0].text


def get_global_news_openai(curr_date, ticker):
    config = get_config()
    client = OpenAI(base_url=config["backend_url"])

    # 计算时间范围
    from datetime import datetime, timedelta
    if isinstance(curr_date, str):
        curr_date = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date = (curr_date - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = curr_date.strftime("%Y-%m-%d")

    prompt = f"""
    You are a professional macroeconomic news extraction assistant.

Your task is to:
1. Identify the company name and main industry sector from the ticker symbol: **{ticker}**
2. Then, extract major macroeconomic and industry-related news between **{start_date} and {end_date}** that may affect this company or its sector globally.


### CRITICAL INSTRUCTIONS:

1. Use reliable financial and economic sources only (e.g., Investing.com, Reuters, WSJ, CNBC, Yahoo Finance, central bank releases).

2. Search in two areas:
#### (A) Global Macro News
Cover only **major developments** in:
   - Central bank policy (interest rates, QE/QT, guidance)
   - Key economic indicators (CPI, GDP, employment, PMI, etc.)
   - Geopolitical events (wars, trade tensions, sanctions)
   - Global sector-wide trends (e.g., oil demand, supply chain, AI chip bans)
#### (B) Sector/Industry-Specific News
After identifying the company and its sector:
   - Regulatory or policy changes affecting the industry
   - Major technological breakthroughs or shifts
   - Cross-border M&A or joint ventures in the industry
   - News involving competitors with systemic impact

2.  For each piece of news found, present it as a single line in a markdown table.
3.  Do NOT add any commentary, analysis, ranking, or opinions. Report the facts only.
4.  If no significant news is found for a category, omit it from the table.
5.  Each news item must be:
   - Real and published between **{start_date} and {end_date}**
   - Summarized in **2 sentence max** with no opinions
   - Verified from credible sources

---

### OUTPUT FORMAT (Markdown Table Only):
| Category | Headline | Summary |
|---|---|---|
| [e.g., Central Banks] | [Headline of the news] | [A brief, factual summary of the news content] |
| [e.g., Economic Data] | [Headline of the news] | [A brief, factual summary of the news content] |
"""

    response = client.responses.create(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        #"text": f"Can you search global or macroeconomics news from 7 days before {curr_date} to {curr_date} that would be informative for trading purposes? Make sure you only get the data posted during that period.",
                        "text": prompt
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[
            {
                "type": "web_search_preview",
                # "user_location": {"type": "approximate"},
                "search_context_size": "medium", #low/medium/high
            }
        ],
        temperature=0.4,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    return response.output[1].content[0].text


def get_fundamentals_openai(ticker, curr_date):
    config = get_config()
    client = OpenAI(base_url=config["backend_url"])

    response = client.responses.create(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        # "text": f"Can you search Fundamental for discussions on the stock company with ticker symbol {ticker} during the month before {curr_date} to the month of {curr_date}? Make sure you only get the data posted during that period. List as a table, with PE/PS/Cash flow/ etc",
                        "text": f"""
You are an automated financial data extraction bot. Your sole task is to use the web search tool to find the most recent, key fundamental data points for the company with ticker symbol {ticker}

IMPORTANT GUIDELINES:
1.  Use these sources in priority: 
   - Yahoo Finance (finance.yahoo.com) > Google Finance (finance.google.com) > Investing (www.investing.com/equities) > MarketWatch (marketwatch.com) > Reuters (reuters.com) > Company's official investor relations page
2.  Use the most reliable source **for each metric individually**, do not restrict to one site.
3.  Ensure all monetary values are in USD and use consistent formatting (e.g., $1.23B, $123.45M, $123.45).
4.  Understand and interpret technical terms flexibly:
   - TTM = trailing twelve months
   - EV/EBITDA = Enterprise Value divided by EBITDA
   - ROE = Return on Equity
   - Insider Net Activity = Insider share activity in past 6 months
   - Top Institutional Holder = Largest institutional shareholder, by ownership %
5. If a metric is unavailable in table form, check:
   - Earnings releases
   - Company filings (10-K, 10-Q)
   - “Statistics”, “Analysis”, or “Financials”
6.  Output the result ONLY as a single, clean markdown table.
7.  Do NOT include any explanation or commentary.
8. Use "N/A" if a value is unavailable. Do not omit rows.
9. Use the following keyword pairs during search, along with the company name or ticker symbol, to locate metrics more precisely:
    - Market Cap: `"market capitalization"`, `"company value"`
    - P/E Ratio: `"price to earnings ratio"`, `"PE (TTM)"`
    - P/S Ratio: `"price to sales ratio"`, `"PS (TTM)"`
    - EV/EBITDA: `"enterprise value to EBITDA"`, `"EV EBITDA multiple"`
    - Gross Margin: `"gross margin percentage TTM"`
    - Operating Margin: `"operating margin TTM"`, `"EBIT margin"`
    - ROE: `"return on equity TTM"`
    - Debt-to-Equity: `"debt to equity ratio MRQ"`, `"financial leverage"`
    - Current Ratio: `"current ratio MRQ"`, `"liquidity ratio"`
    - Revenue Growth: `"revenue growth YoY"`, `"year-over-year revenue"`
    - EPS Growth: `"EPS growth YoY"`, `"earnings per share growth"`
    - Operating Cash Flow: `"operating cash flow TTM"`, `"cash from operations"`
    - Free Cash Flow: `"free cash flow TTM"`, `"FCF"`
    - Dividend Yield: `"dividend yield"`, `"dividend per share"`
    - Insider Activity: `"insider trading last 6 months"`, `"insider buys/sells summary"`
    - Top Institutional Holder: `"largest institutional shareholder"`, `"top institutional owner"`

    Use combined search patterns like:
    - `"NVIDIA free cash flow TTM site:finance.yahoo.com"`
    - `"TSLA EV/EBITDA MarketWatch"`
    - `"AMD ROE TTM site:reuters.com"`

--- 

### Output Format (Markdown Table):
| Category | Metric | Value |
|---|---|---|
| Valuation | Market Cap | [Value] |
| Valuation | P/E Ratio (TTM) | [Value] |
| Valuation | P/S Ratio (TTM) | [Value] |
| Valuation | EV/EBITDA (TTM) | [Value] |
| Profitability | Gross Margin (TTM) | [Value] |
| Profitability | Operating Margin (TTM) | [Value] |
| Profitability | Return on Equity (ROE, TTM) | [Value] |
| Financial Health| Debt-to-Equity Ratio (MRQ) | [Value] |
| Financial Health| Current Ratio (MRQ) | [Value] |
| Growth | Revenue Growth (YoY) | [Value] |
| Growth | EPS Growth (YoY) | [Value] |
| Cash Flow | Operating Cash Flow (TTM) | [Value] |
| Cash Flow | Free Cash Flow (TTM) | [Value] |
| Dividends & Ownership | Dividend Yield | [Value] |
| Dividends & Ownership | Insider Net Activity (6M) | [e.g., -150,000 shares sold] |
| Dividends & Ownership | Top Institutional Holder | [e.g., Vanguard Group (8.5%)] |
"""
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[
            {
                "type": "web_search_preview",
                # "user_location": {"type": "approximate"},
                "search_context_size": "medium",
                #"max_results": 5,
                #"custom_query": "site:reuters.com finance news",
                #"time_range": {"start": "2024-06-01", "end": "2024-06-07"}
            }
        ],
        temperature=0.3,  # Lower temperature for more consistent results
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    return response.output[1].content[0].text


def get_company_profile_openai(ticker):
    config = get_config()
    client = OpenAI(base_url=config["backend_url"])

    prompt = f"""
You are a professional business analyst bot. Your task is to research and present a **clear, detailed, and structured summary** of the business model for the company with the ticker symbol {ticker}.

CRITICAL INSTRUCTIONS:
1. Use the web search tool to find the latest and most accurate business information. Prioritize sources in this order:
   - Company’s official website and investor relations page
   - Yahoo Finance (finance.yahoo.com)
   - Google Finance (google.com/finance)
   - MarketWatch (marketwatch.com)
   - Reuters (reuters.com)
   - Wikipedia (for well-established firms)
2. The output format must follow this exact structure (in Markdown):
    ## Company Overview
        A concise 2-3 sentence description of what the company is and which industry it operates in.

    ## Core Business Segments
        - **Segment 1 Name**: Brief explanation of what this segment does.

        - **Segment 2 Name**: [Only include if applicable]
        (Up to 3 segments. If unknown, summarize main revenue drivers.)

    ## Main Products or Services
        Bullet list of 2–5 key products, services, or platforms the company offers (If applicable).

    ## Revenue Model
        Explain clearly how the company generates revenue.
        Include key channels: e.g., subscription, product sales, licensing, advertising, transactions, data, platform fees, etc.

    ## Customer Base
        Describe the company’s main customer types (e.g., individual consumers, businesses, government).
        Include regions served if relevant.

    ## Business Strategy or Recent Developments (optional)
        Briefly mention major strategic focus, growth areas, or recent pivots if available.
3. You MUST address the following 3 points:
   - The company’s primary business activities
   - Its main products or services
   - Its primary revenue sources and business model
4. Do NOT include:
   - Financial figures or valuation data
   - Stock price commentary or investment advice
   - Marketing or promotional language
5. Maintain a neutral, formal tone using factual business language.
6. If the company has multiple segments, emphasize the **most significant ones**.
7. Use search phrases like:
   - “[Company name] business model”
   - “[Company name] revenue breakdown”
   - “[Company name] what it does”
   - “[Company name] investor relations business overview”
8. When multiple sources provide overlapping or conflicting information, prioritize:
   - The **most recently updated** source.
   - Official company statements (IR, press releases) over third-party summaries.
   - Wikipedia only if cross-referenced with other sources and considered current.

9. If the publication date or last update of a source is visible, prefer content published within the **last 12 months**, especially for business segments or strategic changes.

10. If a source appears outdated or promotional, deprioritize or discard it.
"""

    response = client.responses.create(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[{"type": "web_search_preview", "search_context_size": "medium"}],
        temperature=0.3,
        max_output_tokens=4096,
        top_p=1,
    )

    return response.output[1].content[0].text
