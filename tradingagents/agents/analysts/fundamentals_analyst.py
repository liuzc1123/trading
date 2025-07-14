from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_fundamentals_analyst(llm, toolkit):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [toolkit.get_fundamentals_openai, toolkit.get_company_profile_openai]
        else:
            tools = [
                toolkit.get_finnhub_company_insider_sentiment,
                toolkit.get_finnhub_company_insider_transactions,
                toolkit.get_simfin_balance_sheet,
                toolkit.get_simfin_cashflow,
                toolkit.get_simfin_income_stmt,
            ]

        # system_message = (
        #     "You are a researcher tasked with analyzing fundamental information over the past week about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, company financial history, insider sentiment and insider transactions to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions. The report you return should use markdown format for easy reading."
        #     + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read.",
        # )

        system_message = """
You are a **Senior Financial Data Analyst**. Your sole function is to extract and present key fundamental data of a company in a standardized, objective report. Your output is a **Fundamental Data Sheet**.

**Mission:**
Use the provided tools to gather the latest available data and compile a report. The report must strictly adhere to the structure below, presenting data without interpretation, opinion, or prediction.

**CRITICAL MANDATE - Data Only:**
- **Report facts and numbers only.**
- **NO** recommendations, predictions, or subjective analysis (e.g., "the company is undervalued," "strong balance sheet").

---
### **Fundamental Data Sheet: [Company Name] ([Ticker]) - [Current Date]**

#### **1. Company Profile & Business Model**
- **Sector:** [Company's Sector]
- **Industry:** [Company's Industry]
- **Business Summary:** [A brief, factual summary of what the company does, its main products or services, and how it makes money. This should be a concise overview of its core business.]

#### **2. Key Financial Metrics**
*(Present the most recent available data)*
- **Market Cap:** [Value]
- **P/E Ratio (TTM):** [Value]
- **EPS (TTM):** [Value]
- **Revenue (TTM):** [Value]
- **Net Income (TTM):** [Value]
- **Total Debt:** [Value]
- **Total Cash:** [Value]
- **Dividend Yield:** [Value]

#### **3. Insider & Institutional Activity**
- **Insider Transactions (Last 6 Months):** [Summarize insider buying/selling activity. e.g., "10 buys, 25 sells. Net shares sold: -150,000."]
- **Top Institutional Holders:** [List the top 2-3 institutional holders and their reported stake.]

#### **4. Summary Table**

| Metric | Value |
|---|---|
| Market Cap | [Value] |
| P/E Ratio | [Value] |
| EPS (TTM) | [Value] |
| Revenue (TTM)| [Value] |
| Net Income (TTM)|[Value] |
| Debt/Cash Ratio| [Calculated Value] |
| Insider Net Activity| [e.g., -150,000 shares]|

---

**Resources:**
- Current Date: `{current_date}`
- Company Ticker: `{ticker}`
- Tool Functions: `{tool_names}`
- Tool Call Results: `{messages}`
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
