#  Swarm：openai 提出了用于创建和协调多智能体系统的轻量级技术。
# Swarm 最有趣的想法可能是切换：让一个代理通过工具调用将控制权转移给另一个代理。
# https://colab.research.google.com/github/deepset-ai/haystack-cookbook/blob/main/notebooks/swarm.ipynb#scrollTo=Z8TO6XRaGVYZ

import os
import sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Annotated, Callable, Tuple
from dataclasses import dataclass, field

import random, re

from haystack.utils import ComponentDevice
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.dataclasses import StreamingChunk
from haystack.components.generators.utils import print_streaming_chunk
from haystack.tools import create_tool_from_function
from haystack.components.tools import ToolInvoker
from haystack.components.generators.chat import OpenAIChatGenerator, HuggingFaceLocalChatGenerator
# from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
# from haystack_integrations.components.generators.ollama import OllamaChatGenerator


@dataclass
class Assistant:
    name: str = "Assistant"
    llm: object = HuggingFaceLocalChatGenerator(
        model="Qwen/Qwen3-0.6B",
        task="text-generation",
        device=ComponentDevice.from_str("cpu"),  #
        # device=ComponentDevice.resolve_device(None),
        generation_kwargs={
            "max_new_tokens": 2048,   # input prompt + max_new_tokens / max32768
            "temperature": 0.9,
            "top_p": 0.95,
            "repetition_penalty": 1.2,
        }, 
        # streaming_callback=print_streaming_chunk, 
    )
    llm.warm_up()
    instructions: str = "You are a helpful Agent"

    def __post_init__(self):
        self._system_message = ChatMessage.from_system(self.instructions)

    def run(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        new_message = self.llm.run(messages=[self._system_message] + messages)["replies"][0]

        if new_message.text:
            print(f"\n{self.name}: {new_message.text}")

        return [new_message]


@dataclass
class ToolCallingAgent:
    name: str = "ToolCallingAgent"
    llm: object = HuggingFaceLocalChatGenerator(
        model="Qwen/Qwen3-0.6B",
        task="text-generation",
        device=ComponentDevice.from_str("cpu"),  #
        # device=ComponentDevice.resolve_device(None),
        generation_kwargs={
            "max_new_tokens": 2048,   # input prompt + max_new_tokens / max32768
            "temperature": 0.9,
            "top_p": 0.95,
            "repetition_penalty": 1.2,
        }, 
        # streaming_callback=print_streaming_chunk, 
    )
    llm.warm_up()
    instructions: str = "You are a helpful Agent"
    functions: list[Callable] = field(default_factory=list)

    def __post_init__(self):
        self._system_message = ChatMessage.from_system(self.instructions)
        self.tools = [create_tool_from_function(fun) for fun in self.functions] if self.functions else None
        self._tool_invoker = ToolInvoker(tools=self.tools, raise_on_failure=False) if self.tools else None

    def run(self, messages: list[ChatMessage]) -> Tuple[str, list[ChatMessage]]:

        # generate response
        agent_message = self.llm.run(messages=[self._system_message] + messages, tools=self.tools)["replies"][0]
        new_messages = [agent_message]

        if agent_message.text:
            print(f"\n{self.name}: {agent_message.text}")

        if not agent_message.tool_calls:
            return new_messages

        # handle tool calls
        tool_results = self._tool_invoker.run(messages=[agent_message])["tool_messages"]
        new_messages.extend(tool_results)

        return new_messages


@dataclass
class SwarmAgent:
    name: str = "SwarmAgent"
    llm: object = HuggingFaceLocalChatGenerator(
        model="Qwen/Qwen3-0.6B",
        task="text-generation",
        device=ComponentDevice.from_str("cpu"),  #
        # device=ComponentDevice.resolve_device(None),
        generation_kwargs={
            "max_new_tokens": 2048,   # input prompt + max_new_tokens / max32768
            "temperature": 0.9,
            "top_p": 0.95,
            "repetition_penalty": 1.2,
        }, 
        # streaming_callback=print_streaming_chunk,   # 异步回调
    )
    llm.warm_up()
    instructions: str = "You are a helpful Agent"
    functions: list[Callable] = field(default_factory=list)

    def __post_init__(self):
        self._system_message = ChatMessage.from_system(self.instructions)
        self.tools = [create_tool_from_function(fun) for fun in self.functions] if self.functions else None
        self._tool_invoker = ToolInvoker(tools=self.tools, raise_on_failure=False) if self.tools else None

    def run(self, messages: list[ChatMessage]) -> Tuple[str, list[ChatMessage]]:
        # generate response
        agent_message = self.llm.run(messages=[self._system_message] + messages, tools=self.tools)["replies"][0]
        new_messages = [agent_message]

        if agent_message.text:
            print(f"\n{self.name}: {agent_message.text}")

        if not agent_message.tool_calls:
            return self.name, new_messages

        # handle tool calls
        for tc in agent_message.tool_calls:
            # trick: Ollama does not produce IDs, but OpenAI and Anthropic require them.
            if tc.id is None:
                tc.id = str(random.randint(0, 1000000))
        tool_results = self._tool_invoker.run(messages=[agent_message])["tool_messages"]
        new_messages.extend(tool_results)

        # handoff
        last_result = tool_results[-1].tool_call_result.result
        match = re.search(HANDOFF_PATTERN, last_result)
        new_agent_name = match.group(1) if match else self.name

        return new_agent_name, new_messages


"""
更复杂的多智能体系统，它模拟了 ACME 公司的客户服务设置

该系统涉及几个不同的代理（每个代理都有特定的工具）：
- 分诊代理：处理一般问题并转接给其他代理。工具：transfer_to_sales_agent、transfer_to_issues_and_repairs和escalate_to_human。
- 销售代理：向用户推荐并销售产品，它可以执行订单或将用户重定向回分诊代理。工具：execute_order和transfer_back_to_triage。
- 问题与维修代理：为客户提供问题支持，可以查找商品 ID、执行退款或将用户重定向回分诊处。工具：look_up_item、 execute_refund、 和transfer_back_to_triage。
"""

HANDOFF_TEMPLATE = "Transferred to: {agent_name}. Adopt persona immediately."
HANDOFF_PATTERN = r"Transferred to: (.*?)(?:\.|$)"

def escalate_to_human(summary: Annotated[str, "A summary"]):
    """Only call this if explicitly asked to."""
    print("Escalating to human agent...")
    print("\n=== Escalation Report ===")
    print(f"Summary: {summary}")
    print("=========================\n")
    exit()


def transfer_to_sales_agent():
    """Use for anything sales or buying related."""
    return HANDOFF_TEMPLATE.format(agent_name="Sales Agent")


def transfer_to_issues_and_repairs():
    """Use for issues, repairs, or refunds."""
    return HANDOFF_TEMPLATE.format(agent_name="Issues and Repairs Agent")


def transfer_back_to_triage():
    """Call this if the user brings up a topic outside of your purview,
    including escalating to human."""
    return HANDOFF_TEMPLATE.format(agent_name="Triage Agent")


triage_agent = SwarmAgent(
    name="Triage Agent",
    instructions=(
        "You are a customer service bot for ACME Inc. "
        "Introduce yourself. Always be very brief. "
        "If the user asks general questions, try to answer them yourself without transferring to another agent. "
        "Only if the user has problems with already bought products, transfer to Issues and Repairs Agent."
        "If the user looks for new products, transfer to Sales Agent."
        "Make tool calls only if necessary and make sure to provide the right arguments."
    ),
    functions=[transfer_to_sales_agent, transfer_to_issues_and_repairs, escalate_to_human],
)


def execute_order(
    product: Annotated[str, "The name of the product"], price: Annotated[int, "The price of the product in USD"]
):
    print("\n\n=== Order Summary ===")
    print(f"Product: {product}")
    print(f"Price: ${price}")
    print("=================\n")
    confirm = input("Confirm order? y/n: ").strip().lower()
    if confirm == "y":
        print("Order execution successful!")
        return "Success"
    else:
        print("Order cancelled!")
        return "User cancelled order."


sales_agent = SwarmAgent(
    name="Sales Agent",
    instructions=(
        "You are a sales agent for ACME Inc."
        "Always answer in a sentence or less."
        "Follow the following routine with the user:"
        "1. Ask them about any problems in their life related to catching roadrunners.\n"
        "2. Casually mention one of ACME's crazy made-up products can help.\n"
        " - Don't mention price.\n"
        "3. Once the user is bought in, drop a ridiculous price.\n"
        "4. Only after everything, and if the user says yes, "
        "tell them a crazy caveat and execute their order.\n"
        ""
    ),
    functions=[execute_order, transfer_back_to_triage],
)


def look_up_item(search_query: Annotated[str, "Search query to find item ID; can be a description or keywords"]):
    """Use to find item ID."""
    item_id = "item_132612938"
    print("Found item:", item_id)
    return item_id


def execute_refund(
    item_id: Annotated[str, "The ID of the item to refund"], reason: Annotated[str, "The reason for refund"]
):
    print("\n\n=== Refund Summary ===")
    print(f"Item ID: {item_id}")
    print(f"Reason: {reason}")
    print("=================\n")
    print("Refund execution successful!")
    return "success"


issues_and_repairs_agent = SwarmAgent(
    name="Issues and Repairs Agent",
    instructions=(
        "You are a customer support agent for ACME Inc."
        "Always answer in a sentence or less."
        "Follow the following routine with the user:"
        "1. If the user is interested in buying or general questions, transfer back to Triage Agent.\n"
        "2. First, ask probing questions and understand the user's problem deeper.\n"
        " - unless the user has already provided a reason.\n"
        "3. Propose a fix (make one up).\n"
        "4. ONLY if not satesfied, offer a refund.\n"
        "5. If accepted, search for the ID and then execute refund."
        ""
    ),
    functions=[look_up_item, execute_refund, transfer_back_to_triage],
)

# ======== 

agents = {agent.name: agent for agent in [triage_agent, sales_agent, issues_and_repairs_agent]}

print("Type 'quit' to exit")

messages = []
current_agent_name = "Triage Agent"

while True:
    agent = agents[current_agent_name]

    if not messages or messages[-1].role == ChatRole.ASSISTANT:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            break
        messages.append(ChatMessage.from_user(user_input))

    current_agent_name, new_messages = agent.run(messages)
    messages.extend(new_messages)
