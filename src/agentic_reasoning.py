from typing import Dict, Iterator, List, Literal, Optional

from agno.agent import Agent, RunResponse
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory
from agno.models.openai import OpenAIChat
from agno.team import Team
from agno.tools import tool
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.python import PythonTools
from agno.tools.reasoning import ReasoningTools
from agno.utils.pprint import pprint_run_response
from pydantic import BaseModel, Field

# Create a memory instance with persistent storage
# memory_db = SqliteMemoryDb(table_name="memory", db_file="memory.db")
# memory = Memory(db=memory_db)

def retriever(agent: Agent, query: str, num_documents: Optional[int], **kwargs) -> Optional[list[dict]]:
  pass


valid_tools= ["web_search", "coding", "mind_map", "own_knowledge"]

class SingleReasoningStep(BaseModel):

    reason: str = Field(..., description="A single step in the reasoning chain. This should be as minute and atomic as possible. It should not have dependency with any other reasoning step.")
    tool: Literal[tuple(valid_tools)] = Field(..., description="The tool which is to be used for this particular step of the reasoning chain", enum=valid_tools)

class ReasoningChain(BaseModel):

    summaries: List[SingleReasoningStep]= Field(..., description="A list of reasoning steps")



web_search_agent = Agent(
    name="Web research expert",
    role="Expert at finding information by using online web search to retrieve information of which the agent does not readily have knowledge.",
    tools=[DuckDuckGoTools()],
    model=OpenAIChat("gpt-4o"),
    show_tool_calls=True
)


coding_agent = Agent(
    name="Coding expert",
    role="Expert at writing Python code for solving any problem required.",
    tools=[PythonTools()],
    model=OpenAIChat("gpt-4o"),
    show_tool_calls=True
)

mind_map_agent= Agent(
    name="Mind map agent",
    role="Takes the reasoning chain and the information generated so far to orgainze a mind map for retrieval and reference.",
    tools=[ReasoningTools()],
    model=OpenAIChat("gpt-4o"),
    show_tool_calls=True
)


lead_researcher_system_prompt= """You are an intelligent researcher. Given any query, your goal is to understand it minutely,
break it down into a series of steps required to come to the final conclusion and utilize your team for executing your steps.
"""

success_criteria_prompt="""Produce actionable, verifiable and data backed replies supported by extensive analysis"""

instructions_list= [
"For any given query, think step by step and break down the query into atomic steps which capture the actions you need to carry out to get an answer to the task.",
"Then execute the steps one by one to gather data, understanding etc.",
"Use the mind map agent to structure and organize your reasoning process and any data you have collected. ",
"Finally, use the data you have gathered to come to a final, well-structured answer to the user query.",
    ]

team = Team(
   name= "Lead researcher",
   mode="coordinate",
   model=OpenAIChat(id="gpt-4o"),
    description=lead_researcher_system_prompt,
    instructions=instructions_list,
    success_criteria=success_criteria_prompt,
    tools=[ReasoningTools()],
    members=[web_search_agent, coding_agent,],
    add_datetime_to_instructions= True, # This allows for relative times like "tomorrow" to be used in the prompt
    enable_agentic_context=True,  # Enable Team Leader to maintain Agentic Context
    share_member_interactions=True,  # Share interactions
    enable_team_history=True,
    num_of_interactions_from_history=50,
    show_members_responses=True,
    markdown=True,
    reasoning=True
)



while True:
    query= input("Please enter your query: ")
    team.print_response(message= query,
                        stream= True,
                        stream_intermediate_steps= True,
                        show_reasoning= True,
                        show_reasoning_verbose= True
                        )
