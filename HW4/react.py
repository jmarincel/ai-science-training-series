from typing import TypedDict, Annotated

from langgraph.graph import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from inference_auth_token import get_access_token

from tools import lookup_on_arxiv#, download_arxiv_paper


# ============================================================
# 1. State definition
# ============================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# 2. Routing logic
# ============================================================
def route_tools(state: State):
    """Route to the 'tools' node if the last message has tool calls; otherwise, route to 'done'.

    Parameters
    ----------
    state : State
        The current state containing messages and remaining steps

    Returns
    -------
    str
        Either 'tools' or 'done' based on the state conditions
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "done"


# ============================================================
# 3. LLM node: the "agent"
# ============================================================
def arxiv_agent(
    state: State,
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str = "You are an assistant that uses tools to solve problems ",
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    llm_with_tools = llm.bind_tools(tools=tools)
    return {"messages": [llm_with_tools.invoke(messages)]}

# ============================================================
# 3*. A second agent: Handle creating structured output
# ============================================================


def structured_output_agent(
    state: State,
    llm: ChatOpenAI,
    system_prompt: str = ("You create a table with ONLY 3 ROWS AND 2 COLUMNS: title and hyperlink. "),
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]

    result = llm.invoke(messages)
    return {"messages": [result]}


# ============================================================
# 4. LLM / tools setup
# ============================================================
# Get token for your ALCF inference endpoint
access_token = get_access_token()

# Initialize the model hosted on the ALCF endpoint
llm = ChatOpenAI(
    model_name="openai/gpt-oss-120b",
    api_key=access_token,
    base_url="https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1",
    temperature=0,
)

# Tool list that the LLM can call
tools = [lookup_on_arxiv]

# ============================================================
# 5. Build the graph
# ============================================================
graph_builder = StateGraph(State)

# Agent node: calls LLM, which may decide to call tools
graph_builder.add_node(
    "arxiv_agent",
    lambda state: arxiv_agent(state, llm=llm, tools=tools),
)
graph_builder.add_node(
    "structured_output_agent",
    lambda state: structured_output_agent(state, llm=llm),
)


# Tool node: executes tool calls emitted by the LLM
tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)

# Graph logic
# START -> arxiv_agent
graph_builder.add_edge(START, "arxiv_agent")

# After chem_agent runs, check if we need to run tools
graph_builder.add_conditional_edges("arxiv_agent", route_tools, {"tools": "tools", "done": "structured_output_agent"})

# After tools run, go back to the agent so it can use tool results
graph_builder.add_edge("tools", "arxiv_agent")

# After structured_output_agent, terminate the graph
graph_builder.add_edge("structured_output_agent", END)

# Compile the graph
graph = graph_builder.compile()

# ============================================================
# 6. Run / stream the graph
# ============================================================
prompt = (
    "What are some papers on the arxiv exploring the possibility of a black to white hole transition?"
)
for chunk in graph.stream(
    {"messages": prompt},
    stream_mode="values",
):
    new_message = chunk["messages"][-1]
    new_message.pretty_print()