# https://haystack.deepset.ai/tutorials/43_building_a_tool_calling_agent
# https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/43_Building_a_Tool_Calling_Agent.ipynb

import os
import sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from haystack.components.agents import Agent
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
from haystack.components.websearch.searchapi import SearchApiWebSearch
from haystack.dataclasses import ChatMessage
from haystack.tools.component_tool import ComponentTool

from typing import Annotated, Literal
from haystack.tools import create_tool_from_function    #
from uniqa.tools import Tool
from uniqa.tools.tool_invoker import ToolInvoker

import os
os.environ["SEARCHAPI_API_KEY"] = "Zie7hh93PcACAdBp2sTQSzgP"
websearch = SearchApiWebSearch()

from haystack.utils import ComponentDevice
generator = HuggingFaceLocalChatGenerator(
    model="Qwen/Qwen3-0.6B",    # "Qwen/Qwen2.5-0.5B"
    task="text-generation",
    device=ComponentDevice.from_str("cpu"),  #
    # device=ComponentDevice.resolve_device(None),
    generation_kwargs={
        "max_new_tokens": 32768,   # input prompt + max_new_tokens
        "temperature": 0.9,
        "top_p": 0.95,
    }
)
generator.warm_up()

# Create a web search tool using SerperDevWebSearch
web_tool = ComponentTool(component=websearch, name="web_tool")


def get_weather(
    city: Annotated[str, "the city for which to get the weather"] = "Berlin",
    unit: Annotated[Literal["Celsius", "Fahrenheit"], "the unit for the temperature"] = "Celsius",
):
    """A simple function to get the current weather for a location."""
    WEATHER_INFO = {
        "Berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
        "Paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
        "Rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
        "Madrid": {"weather": "sunny", "temperature": 10, "unit": "celsius"},
        "London": {"weather": "cloudy", "temperature": 9, "unit": "celsius"},
    }

    if city in WEATHER_INFO:
        return WEATHER_INFO[city]
    else:
        return {"weather": "sunny", "temperature": 21.8, "unit": "fahrenheit"}


# 函数封装成工具类 Tool
weather_tool = create_tool_from_function(get_weather)

# Create the agent with the web search tool
agent = Agent(
    chat_generator=generator, 
    tools=[web_tool, weather_tool], 
    system_prompt="""你是一个有用的助手。你的任务是帮助用户解决各种问题。""",
    exit_conditions=["text"],    # List of conditions that will cause the agent to return.
    max_agent_steps=2            # Maximum number of steps the agent will run before stopping.
)
agent.warm_up()

# Run the agent with a query
result = agent.run(messages=[ChatMessage.from_user("Find information about Haystack AI framework")])

# Print the final response
print(result["messages"])
print(result["messages"][-1].text)

