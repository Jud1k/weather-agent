import os
from dotenv import load_dotenv
from fastmcp import FastMCP
import httpx
from pydantic import BaseModel, ConfigDict

mcp = FastMCP(
    name="TravelAssistantServer",
    instructions="""This server provides data analysis tools
    Call get_weather() to get information about weather in city
    """,
)

load_dotenv()


class WeatherResponse(BaseModel):
    model_config=ConfigDict()
    
    city: str
    tempeature: float


@mcp.tool
async def get_weather(city_name: str):
    """
    Get current weather for city
    Args:
        city_name (str): Name of the city

    Returns:
        String with information about weather or error
    """

    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return "Error: can't find API KEY"
    url = "https://api.weatherapi.com/v1/current.json"

    params = {"q": city_name, "key": api_key}
    headers = {"accept": "application/json"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url=url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            current = data["current"]
            location = data["location"]

            return (
                f"Weather in city {location["name"]}\n"
                f"Tempeature: {current["temp_c"]}°C\n"
                f"Wind: {current["wind_kph"]} Kph\n"
                f"Condition: {current["condition"]["text"]}\n"
                f"Feel's like: {current["feelslike_c"]}°C\n"
            )
    except httpx.HTTPError as e:
        return f"Error: {str(e)}"
    except KeyError as e:
        return f"Error in format data from API: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


if __name__ == "__main__":
    try:
        mcp.run(transport="http", host="127.0.0.1", port="9000")
    except KeyboardInterrupt:
        print("Server stoped...")