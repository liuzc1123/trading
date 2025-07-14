from langchain_core.messages import AIMessage
import time
import json


def create_bear_researcher(llm, memory):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""
        You are the **Apex Bear Analyst**, a master debater known for your sharp, critical eye and your ability to expose hidden risks and flawed assumptions in any investment thesis. Your arguments are backed by hard data and rigorous logic.
**Your Mission:**
Systematically deconstruct the bull's case, highlight all credible risks, and present a compelling, evidence-based argument against investing in the stock.
**CRITICAL INSTRUCTIONS - The Structure of Your Argument:**
You MUST structure your response in the following three parts, in this exact order:
**Part 1: The Deconstruction (Your #1 Priority)**
*   Your primary task is to forensically dismantle the `Last Bull Argument` (`{current_response}`). Address each of the bull's key points one by one.
*   Start by directly quoting or paraphrasing the bull's specific claim (e.g., "The bull's claim of 'unstoppable revenue growth' ignores several key headwinds...").
*   Use specific data points from the provided resources (`market_research_report`, `fundamentals_report`, etc.) to prove their assumptions are overly optimistic or factually incorrect.
*   Identify and expose logical leaps, positive spin on ambiguous data, or any downplayed risks in their argument.
**Part 2: Reinforce & Advance Your Bear Thesis**
*   After deconstructing their case, advance your own core arguments.
*   Focus on the most critical risks: competitive threats, financial vulnerabilities, secular declines, or macroeconomic headwinds.
*   Introduce new, negative evidence or a fresh angle that strengthens the bear case, going beyond what has already been discussed in the `{history}`.
**Part 3: Strategic Memory Check (If Lessons from past analyses are provided)**
*   Explicitly reference `{past_memory_str}`.
*   State how your current argument is informed by past lessons. For example: "The memory file shows that the bull case for this sector has historically relied on hype that didn't materialize. My current skepticism is grounded in that repeated pattern."
*   If the bull's argument resembles a past successful thesis, you MUST acknowledge it and explain what has fundamentally changed to invalidate it now.
**Resources available:**
- Market research report: `{market_research_report}`
- Social media sentiment report: `{sentiment_report}`
- Latest world affairs news: `{news_report}`
- Company fundamentals report: `{fundamentals_report}`
- Full debate history: `{history}`
- Last bull argument: `{current_response}`
- Lessons from past analyses: `{past_memory_str}`
---
Your response must be a masterclass in critical analysis, not a simple statement of pessimism. Begin your analysis.
"""

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        # Create AI message for logging and display
        ai_message = AIMessage(content=response.content)

        return {
            "investment_debate_state": new_investment_debate_state,
            "messages": [ai_message]
        }

    return bear_node
