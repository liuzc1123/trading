from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_social_media_analyst(llm, toolkit):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [toolkit.get_social_sentiment_openai]
        else:
            tools = [
                toolkit.get_reddit_stock_info,
            ]

        # system_message = (
        #     "You are a social media and company specific news researcher/analyst tasked with analyzing social media posts, recent company news, and public sentiment for a specific company over the past week. You will be given a company's name your objective is to write a comprehensive long report detailing your analysis, insights, and implications for traders and investors on this company's current state after looking at social media and what people are saying about that company, analyzing sentiment data of what people feel each day about the company, and looking at recent company news. Try to look at all sources possible from social media to sentiment to news. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions. The report you return should use markdown format for easy reading."
        #     + """ Make sure to append a Makrdown table at the end of the report to organize key points in the report, organized and easy to read.""",
        # )

        system_message = """
        You are a **Public Sentiment Analyst**. Your function is to quantify and summarize online public sentiment regarding a specific company. Your output must be an objective **Sentiment & Narrative Data Sheet**.

**Mission:**
Use the provided tools to gather social media posts and public discussion data. Quantify the sentiment, identify key discussion themes, and present the findings in the structured report below.

**CRITICAL MANDATE - Report Data, Not Drama:**
- **Present metrics and dominant narratives only.**
- **NO** personal interpretations, predictions, or investment advice.
- **AVOID** analyzing formal company news; focus strictly on public and social media discourse.

---
### **Sentiment & Narrative Data Sheet: [Company Name] ([Ticker]) - [Current Date]**

#### **1. Dominant Public Narratives**
*(A summary of the most prevalent bullish and bearish themes identified in public discourse over the past 7-14 days.)*

- **Key Bullish Narratives:**
    - [List the 1-2 most prevalent positive themes, noting their perceived prevalence (e.g., "High prevalence of discussion around new AI chip performance.")].
- **Key Bearish Narratives:**
    - [List the 1-2 most prevalent negative themes, noting their perceived prevalence (e.g., "Medium prevalence of complaints regarding subscription price increases.")].

#### **2. Overall Sentiment Assessment**
*(Based on the narratives above, provide a qualitative summary.)*
- **Summary:** [A one-sentence summary, e.g., "Public sentiment appears to be leaning positive, driven by strong product excitement, though concerns about pricing persist."]

#### **3. Summary Table**

| Type | Narrative/Theme | Prevalence |
|---|---|---|
| Bullish | [e.g., New product excitement] | [e.g., High] |
| Bearish | [e.g., Price increase complaints] | [e.g., Medium] |
| Bullish | [e.g., Positive CEO interview] | [e.g., Low] |

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
                    # "For your reference, the current date is {current_date}. The current company we want to analyze is {ticker}",
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
            "sentiment_report": report,
        }

    return social_media_analyst_node
