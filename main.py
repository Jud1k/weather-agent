import asyncio
import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from fastmcp import Client
from langgraph.graph import StateGraph, END
from langchain_gigachat import GigaChat
from langchain.prompts import PromptTemplate
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import BaseMessage, HumanMessage

from utils import generate_graph_png

load_dotenv()

mcp_client = Client(os.getenv("MCP_URL"))


class State(BaseModel):
    """State of the agent for storing information about city"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: List[BaseMessage] = Field(default_factory=list)
    city_name: str = Field("", description="Name of city")


class WeatherAgent:
    """Async agent to receive weather for city"""

    def __init__(self, model: LanguageModelLike):
        """Initialize agent"""
        self.model = model
        self.workflow = self._init_workflow()

    def _init_workflow(self) -> StateGraph:
        """Create workflow agent on basic LangGraph"""
        workflow = StateGraph(State)

        # Add nodes to graph
        workflow.add_node("city_name_classification", self._classify_city_name)
        workflow.add_node("weather_fetching", self._get_weather)

        # Add edges to graph
        workflow.set_entry_point("city_name_classification")
        workflow.add_edge("city_name_classification", "weather_fetching")
        workflow.add_edge("weather_fetching", END)

        app = workflow.compile()
        generate_graph_png(app)
        return app

    async def _classify_city_name(self, state: State) -> Dict[str, Any]:
        """Node to classify city name from prompt"""
        if not state.messages:
            return {
                "city_name": "",
                "messages": [HumanMessage(content="No message provided")],
            }

        last_message = state.messages[-1].content

        prompt = PromptTemplate(
            input_variables=["message"],
            template="""
            Analyze message and define name of the city. Return ONLY city name.

            Message: {message}
            
            You need take only 1 city. Return only city name, nothing else.
            """,
        )

        message = HumanMessage(content=prompt.format(message=last_message))
        response = await self.model.ainvoke([message])
        city_name = response.content.strip()

        return {"city_name": city_name, "messages": state.messages}

    async def _get_weather(self, state: State) -> Dict[str, Any]:
        """Node to get weather for the city"""
        if not state.city_name:
            return {
                "messages": [HumanMessage(content="City not specified")],
                "city_name": "",
            }

        try:
            async with mcp_client as client:
                result = await client.call_tool(
                    "get_weather", arguments={"city_name": state.city_name}
                )

                data = result.content[0].text if result.content else "No weather data"

                response_message = HumanMessage(content=data)

                return {"messages": [response_message], "city_name": state.city_name}

        except Exception as e:
            error_message = HumanMessage(content=f"Error while getting data: {e}")
            return {"messages": [error_message], "city_name": state.city_name}

    async def get_weather(self, user_input: str):
        """Main method to get weather"""
        initial_state = State(messages=[HumanMessage(content=user_input)], city_name="")
        result = await self.workflow.ainvoke(initial_state)
        return result


async def main():
    model = GigaChat(
        credentials=os.getenv("GIGACHAT_API_KEY"),
        verify_ssl_certs=False,
        temperature=0,
    )

    agent = WeatherAgent(model=model)

    print("Weather Agent started!")

    while True:
        try:
            user_input = input("Пользователь: ").strip()

            result = await agent.get_weather(user_input)

            if result and result.get("messages"):
                last_message = result["messages"][-1]
                print("GigaChat:", last_message.content)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
