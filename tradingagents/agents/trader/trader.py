import functools
import json


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        # 当前情境：用于 memory 检索（若 memory 暂未实现可忽略）
        curr_situation = (
            f"Market Research:\n{market_research_report}\n\n"
            f"Sentiment:\n{sentiment_report}\n\n"
            f"News:\n{news_report}\n\n"
            f"Fundamentals:\n{fundamentals_report}"
        )
        past_memories = memory.get_memories(curr_situation, n_matches=2) if memory else []

        if past_memories:
            past_memory_str = "\n\n".join(rec.get("recommendation", "") for rec in past_memories)
        else:
            past_memory_str = "No relevant past memories."

        # ============ Prompt ============
        system_prompt = (
            "You are a professional trading agent responsible for turning research outputs into actionable "
            "trading instructions. Analyse the provided information and output a concise, structured decision.\n\n"
            "You MUST respond strictly in the following JSON format:\n"
            "```json\n"
            "{\n"
            "  \"decision\": \"BUY|HOLD|SELL\",\n"
            "  \"confidence\": 0-100,                 /* integer */\n"
            "  \"rationale\": \"<max 520 words>\",\n"
            "  \"execution_plan\": \"<max 280 words>\"\n"
            "}\n"
            "```\n\n"
            "After the JSON block, append exactly one line in the form:\n"
            "FINAL TRANSACTION PROPOSAL: **<BUY/HOLD/SELL>**\n\n"
            "Leverage the \"Past Reflections\" section to avoid repeating previous mistakes."
        )

        user_prompt = (
            f"### Company of interest\n{company_name}\n\n"
            f"### Proposed investment plan (research team)\n{investment_plan}\n\n"
            f"### Current market snapshot (trimmed)\n"
            f"- Market: {market_research_report[:500]}...\n"
            f"- Sentiment: {sentiment_report[:500]}...\n"
            f"- News: {news_report[:500]}...\n"
            f"- Fundamentals: {fundamentals_report[:500]}...\n\n"
            f"### Past reflections\n{past_memory_str}"
        )
        # context = {
        #     "role": "user",
        #     "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan 
        #     tailored for {company_name}. This plan incorporates insights from current technical market trends, 
        #     macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating 
        #     your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nLeverage these insights 
        #     to make an informed and strategic decision.",
        # }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            # {
            #     "role": "system",
            #     "content": f"""You are a trading agent analyzing market data to make investment decisions. Based 
            #     on your analysis, provide a specific recommendation to buy, sell, or hold. End with a firm 
            #     decision and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' to 
            #     confirm your recommendation. Do not forget to utilize lessons from past decisions to learn from 
            #     your mistakes. Here is some reflections from similar situatiosn you traded in and the lessons 
            #     learned: {past_memory_str}""",
            # },
            # context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
