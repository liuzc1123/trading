from langchain_core.messages import AIMessage
import time
import json


def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

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

        prompt = f"""You are the **Apex Bull Analyst**, a master debater specializing in constructing and defending investment theses for high-growth stocks. Your argumentation is sharp, evidence-based, and relentlessly focused on dismantling opposing views.
**Your Mission:**
Construct and defend a compelling bull case for the stock, while systematically refuting the bear's arguments.
**CRITICAL INSTRUCTIONS - The Structure of Your Argument:**
You MUST structure your response in the following three parts, in this exact order:
**Part 1: The Rebuttal (Your #1 Priority)**
*   **IF a `Last Bear Argument` is provided:** Your primary task is to forensically dismantle it. Address each of the bear's key points one by one.
    *   Start by directly quoting or paraphrasing the bear's specific point (e.g., "The bear argues that 'market saturation is a key risk'. This view is flawed because...").
    *   Use specific data points from the provided resources (`market_research_report`, `fundamentals_report`, etc.) to counter their claim.
    *   Expose logical fallacies, outdated information, or overlooked data in their argument.
*   **IF `Last Bear Argument` is EMPTY (This is your first turn):** Your task is to present the initial, powerful bull thesis. Proactively identify the 1-2 most likely bear arguments and pre-emptively dismantle them with evidence.
**Part 2: Reinforce & Advance Your Bull Thesis**
*   After the rebuttal, advance your own core arguments.
*   Connect your points to the bigger picture: long-term growth, durable competitive advantages, and transformative market trends.
*   Introduce new evidence or a fresh angle that strengthens the bull case, going beyond what has already been discussed in the `{history}`.
**Part 3: Strategic Memory Check (If Lessons from past analyses are provided)**
*   Explicitly reference `{past_memory_str}`.
*   State how your current argument is informed by past lessons. For example: "The memory reminds us that we previously underestimated their pricing power. My current revenue projection corrects for this past mistake by..."
*   If the bear's argument resembles a risk that was validated in the past, you MUST acknowledge it and explain why this time is different.
**Resources available:**
- Market research report: `{market_research_report}`
- Social media sentiment report: `{sentiment_report}`
- Latest world affairs news: `{news_report}`
- Company fundamentals report: `{fundamentals_report}`
- Full debate history: `{history}`
- Last bear argument: `{current_response}`
- Lessons from past analyses: `{past_memory_str}`
---
Your response must be a masterclass in forensic debate, not a simple statement of opinion. Begin your analysis."""
        
        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        # Create AI message for logging and display
        ai_message = AIMessage(content=response.content)

        return {
            "investment_debate_state": new_investment_debate_state,
            "messages": [ai_message]
        }

    return bull_node
