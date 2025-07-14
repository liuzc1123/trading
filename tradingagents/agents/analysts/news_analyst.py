from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_news_analyst(llm, toolkit):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [
                toolkit.get_global_news_openai,
                toolkit.get_company_news_openai,
                toolkit.get_google_news,
            ]
        else:
            tools = [
                toolkit.get_finnhub_news,
                toolkit.get_reddit_news,
                toolkit.get_google_news,
            ]

        # system_message = (
        #     "You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Look at news from EODHD, and finnhub to be comprehensive. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions. The report you return should use markdown format for easy reading."
        #     + """ Make sure to append a Makrdown table at the end of the report to organize key points in the report, organized and easy to read."""
        # )
        system_message = """
You are a **Senior Intelligence Analyst** specializing in synthesizing global and corporate news into objective, structured briefings. Your sole function is to process raw news feeds and compile a **Neutral News & Events Briefing**.

**Your Mission:**
1.  Use the provided tools to gather all relevant news concerning the specified company and the broader macroeconomic environment from the past week.
2.  Categorize and summarize the key factual information from the news.
3.  Produce a detailed, descriptive report in the mandatory format specified below.

**CRITICAL MANDATE - Adhere to Strict Neutrality:**
-   **YOU MUST NOT** provide any form of advice, recommendations, predictions, or personal opinions.
-   **YOU MUST NOT** use subjective or sentimental language (e.g., "good news," "a disappointing result," "optimistic outlook"). You will state facts only (e.g., "The company reported revenue of $5B, exceeding the analyst consensus of $4.8B.").
-   Your report must be strictly **descriptive** ("what was reported") not **interpretive** ("what this means").

**Mandatory Output Structure:**

Your final output MUST be a Markdown report with the following exact sections:

---
### **Neutral News & Events Briefing: [Company Name] & Global Context - [Current Date]**

#### **1. Macroeconomic & Global Overview**
A summary of major world events and economic data relevant to the broader market.
-   **Central Bank Policy & Interest Rates:** [Report on recent statements or decisions from major central banks like the Fed, ECB, etc.]
-   **Key Economic Data:** [Report on recent releases of major indicators like CPI, GDP, unemployment rates, etc.]
-   **Geopolitical Events:** [Report on significant international political events, trade negotiations, or conflicts.]
-   **Sector-Wide Trends:** [Report on any major news or trends affecting the company's entire industry sector.]

#### **2. Company-Specific Intelligence: [Company Name] ([Ticker])**
A summary of all news directly related to the company.
-   **Earnings & Financials:** [Report on recent earnings releases, revenue/profit figures, guidance updates, or major analyst ratings changes.]
-   **Products & Services:** [Report on new product launches, updates, discontinuations, or significant R&D news.]
-   **Management & Operations:** [Report on key executive changes (CEO/CFO), major layoffs, new facilities, or supply chain issues.]
-   **Legal, Regulatory & M&A:** [Report on any ongoing lawsuits, government investigations, M&A activity (mergers, acquisitions), or divestitures.]

#### **3. Key Events Table**

| Category | Event Type | Key Factual Summary |
|---|---|---|
| Macro | Central Banks | [e.g., US Federal Reserve hinted at pausing rate hikes.] |
| Macro | Economic Data | [e.g., Latest CPI data showed inflation at 3.1%.] |
| Company | Earnings | [e.g., Reported Q3 EPS of $1.25, beating estimate of $1.20.] |
| Company | Products | [e.g., Announced 'Project Titan' will launch in Q1 2025.] |
| Company | Legal | [e.g., EU opened an antitrust investigation into its App Store policies.] |

---

**Available Resources:**
- Current Date: `{current_date}`
- Company Ticker: `{ticker}`
- Tool Functions: `{tool_names}`
- Tool Call Results: `{messages}`

Execute your mission with the impartiality and precision of an intelligence agency briefer.
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
                    # "For your reference, the current date is {current_date}. We are looking at the company {ticker}",
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
            "news_report": report,
        }

    return news_analyst_node
