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

import os
os.environ["SEARCHAPI_API_KEY"] = "Zie7hh93PcACAdBp2sTQSzgP"
websearch = SearchApiWebSearch()

from haystack.utils import ComponentDevice
generator = HuggingFaceLocalChatGenerator(
    model="Qwen/Qwen3-0.6B",
    task="text-generation",
    device=ComponentDevice.from_str("cpu"),  #
    # device=ComponentDevice.resolve_device(None),
    generation_kwargs={
        "max_new_tokens": 512,   # input prompt + max_new_tokens
        "temperature": 0.9,
        "top_p": 0.95,
    }
)
generator.warm_up()

# Create a web search tool using SerperDevWebSearch
web_tool = ComponentTool(component=websearch, name="web_tool")

# Create the agent with the web search tool
agent = Agent(chat_generator=generator, tools=[web_tool])
agent.warm_up()

# Run the agent with a query
result = agent.run(messages=[ChatMessage.from_user("Find information about Haystack AI framework")])

# Print the final response
print(result["messages"])
print(result["messages"][-1].text)

