import asyncio
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from playwright.async_api import async_playwright

async def main():
    # ✅ Start Playwright directly
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    tools = toolkit.get_tools()

    # ✅ LLM setup
    llm = ChatOllama(model="gemma3:1b", base_url="http://localhost:11434")

    # ✅ Tool mapping
    tool_by_name = {tool.name: tool for tool in tools}
    navigate_tool = tool_by_name["navigate_browser"]
    get_elements = tool_by_name["get_elements"]

    # ✅ Navigate
    await navigate_tool.arun({"url": "https://www.w3schools.com/html/html_tables.asp"})

    # ✅ Extract table data
    elements = await get_elements.arun({"selector": "td", "attribute": "innerText"})
    print("Extracted <td> contents:", elements)

    # ✅ Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # ✅ Ask agent
    query = "Extract me all Countrys from https://www.w3schools.com/html/html_tables.asp page"
    result = await agent.arun(query)
    print("Agent result:", result)

    # ✅ Cleanup
    await browser.close()
    await playwright.stop()

if __name__ == "__main__":
    asyncio.run(main())