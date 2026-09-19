from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition


def build_agent_graph(llm_with_tools, tools):
    """ReAct-style loop: agent calls LLM, routes to tool execution if requested, repeats until a final answer."""
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", lambda state: {"messages": [llm_with_tools.invoke(state["messages"])]})
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()
