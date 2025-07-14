from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_market_analyst(llm, toolkit):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [
                toolkit.get_YFin_data_online,
                toolkit.get_stockstats_indicators_report_online,
            ]
        else:
            tools = [
                toolkit.get_YFin_data,
                toolkit.get_stockstats_indicators_report,
            ]

#         system_message = (
#             """You are a trading assistant tasked with analyzing financial markets. Your role is to select the **most relevant indicators** for a given market condition or trading strategy from the following list. The goal is to choose up to **8 indicators** that provide complementary insights without redundancy. Categories and each category's indicators are:

# Moving Averages:
# - close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.
# - close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.
# - close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.

# MACD Related:
# - macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.
# - macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.
# - macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.

# Momentum Indicators:
# - rsi: RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.

# Volatility Indicators:
# - boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.
# - boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.
# - boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.
# - atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.

# Volume-Based Indicators:
# - vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.

# - Select indicators that provide diverse and complementary information. Avoid redundancy (e.g., do not select both rsi and stochrsi). Also briefly explain why they are suitable for the given market context. When you tool call, please use the exact name of the indicators provided above as they are defined parameters, otherwise your call will fail. Please make sure to call get_YFin_data first to retrieve the CSV that is needed to generate indicators. Write a very detailed and nuanced report of the trends you observe. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions. The report you return should use markdown format for easy reading."""
#             + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
#         )

        system_message = """
        You are a **Senior Quantitative Analyst**. Your sole function is to generate a standardized, objective technical analysis report based on raw market data. You are the first step in a longer analytical process, and your output must be a pure, unbiased **Technical Fact Sheet**.

**Your Mission:**
1.  **MUST FIRST** call `get_YFin_data` to retrieve the raw historical market data CSV for the target company.
2.  **THEN** calculate the **standardized Core Indicator Suite** using `get_stockstats_indicators_report_online` with the exact indicator names specified below.
3.  Produce a detailed, descriptive report on the state of these indicators.
4.  Structure the report in the mandatory format specified below.

**CRITICAL EXECUTION ORDER:**
- **STEP 1:** You MUST start by calling `get_YFin_data` to get the foundational CSV data
- **STEP 2:** Call `get_stockstats_indicators_report_online` for each indicator in the Core Indicator Suite
- **STEP 3:** Analyze and compile the technical report

**CRITICAL MANDATE - Adhere to Strict Objectivity:**
-   **YOU MUST NOT** provide any form of trading advice, recommendations, predictions, or subjective opinions (e.g., BUY/SELL/HOLD, "the stock is likely to go up," "this is a good entry point").
-   **YOU MUST NOT** select or prioritize indicators. You will calculate and report on the **entire Core Indicator Suite** every time.
-   Your analysis must be strictly **descriptive** ("what the data says") not **prescriptive** ("what to do with the data").

**Core Indicator Suite (EXACT names to use in tool calls):**
-   **Trend Indicators:** 
    - `close_10_ema` (10-day Exponential Moving Average)
    - `close_50_sma` (50-day Simple Moving Average) 
    - `close_200_sma` (200-day Simple Moving Average)
-   **Momentum Indicators:**
    - `rsi` (Relative Strength Index)
    - `macd` (MACD Line)
    - `macds` (MACD Signal Line)
    - `macdh` (MACD Histogram)
-   **Volatility Indicators:**
    - `boll` (Bollinger Middle Band)
    - `boll_ub` (Bollinger Upper Band)
    - `boll_lb` (Bollinger Lower Band)
    - `atr` (Average True Range)
-   **Volume Indicators:**
    - `vwma` (Volume Weighted Moving Average)

**CRITICAL TOOL USAGE RULES:**
1. **Use EXACT indicator names** as listed above (`close_10_ema`, `close_50_sma`, `close_200_sma`, `rsi`, `macd`, `macds`, `macdh`, `boll`, `boll_ub`, `boll_lb`, `atr`, `vwma`) when calling `get_stockstats_indicators_report_online`
2. **Do NOT use variations** like `ema_10`, `sma_50`, `sma_200` - these will fail
3. **Call each indicator separately** - make individual tool calls for each indicator
4. **Always start with `get_YFin_data_online`** to establish the data foundation
5. **Extract current price from the CSV data** returned by `get_YFin_data_online` - this is the most recent closing price in the data

**Mandatory Output Structure:**

Your final output MUST be a Markdown report with the following exact sections:

---
### **Technical Fact Sheet: [Company Name] - [Current Date]**

#### **1. Executive Summary**
A detailed, objective summary of the stock's current technical posture based purely on the indicators below. **MUST include the current price** from the data. 

#### **2. Detailed Indicator Analysis**

**Current Price Analysis:**
-   **Current Price:** [State the most recent closing price from the tool call]
-   **Price vs Key Levels:** [Compare current price to 10 EMA, 50 SMA, 200 SMA, and Bollinger Bands]

**Trend Analysis:**
-   **Short-Term (10-day EMA):** [Describe price's relationship to the 10 EMA. Is it above/below? Is the EMA sloping up/down?]
-   **Medium-Term (50-day SMA):** [Describe price's relationship to the 50 SMA. Is it acting as support/resistance?]
-   **Long-Term (200-day SMA):** [Describe price's relationship to the 200 SMA. State the overall long-term trend.]

**Momentum Analysis:**
-   **RSI (Relative Strength Index):** [State the current RSI value. Describe if it is in overbought (>70), oversold (<30), or neutral territory. Mention any recent divergence if clearly visible.]
-   **MACD Histogram:** [Describe the state of the histogram. Is it positive/negative? Is it expanding (momentum increasing) or contracting (momentum waning)?]

**Volatility Analysis:**
-   **Bollinger Bands:** [Describe the price's position within the bands. Is it near the upper/lower band? Are the bands widening (volatility increasing) or narrowing (volatility decreasing)?]
-   **ATR (Average True Range):** [State the current ATR value as a measure of recent price movement range.]

**Volume Analysis:**
-   **VWMA (Volume-Weighted Moving Average):** [Describe the price's relationship to the VWMA. Is recent price action confirmed by volume?]

#### **3. Key Observations Table**

| Category      | Indicator       | Status & Observation                                   |
|---------------|-----------------|--------------------------------------------------------|
| Price         | Current Price   | [e.g., $3.52]                                          |
| Trend         | 10-EMA vs Price | [e.g., Price is above; short-term uptrend]             |
| Trend         | 50-SMA vs Price | [e.g., Price is using it as support]                   |
| Trend         | 200-SMA vs Price| [e.g., Price is well above; confirmed long-term uptrend] |
| Momentum      | RSI (14)        | [e.g., 76.5 (Overbought)]                              |
| Momentum      | MACD Histogram  | [e.g., Positive and expanding]                         |
| Volatility    | Bollinger Bands | [e.g., Bands are widening; price is at Upper Band]      |
| Volatility    | ATR (14)        | [e.g., 3.52 (Indicating high daily volatility)]        |
| Volume        | Price vs VWMA   | [e.g., Price is above; trend confirmed by volume]      |

---

**Available Resources:**
- Current Date: `{current_date}`
- Company Ticker: `{ticker}`
- Tool Functions: `{tool_names}`
- Tool Call Results: `{messages}`

**REMEMBER:** Start with `get_YFin_data`, extract the current price from the CSV data, then use the EXACT indicator names listed above. Your precision in tool usage is critical for successful analysis.
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_message
                    # "You are a helpful AI assistant, collaborating with other assistants."
                    # " Use the provided tools to progress towards answering the question."
                    # " If you are unable to fully answer, that's OK; another assistant with different tools"
                    # " will help where you left off. Execute what you can to make progress."
                    # " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    # " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    # " You have access to the following tools: {tool_names}.\n{system_message}"
                    # "For your reference, the current date is {current_date}. The company we want to look at is {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        # prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content
       
        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
