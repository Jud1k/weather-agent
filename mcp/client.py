import os
import asyncio
from dotenv import load_dotenv
from fastmcp import Client

load_dotenv()

client = Client(os.getenv("MCP_URL"))


async def test_weather_server():
    async with client:
        try:
            # Basic server interaction
            await client.ping()
            print("✅ Сервер работает!\n")

            # List available operations
            tools = await client.list_tools()
            resources = await client.list_resources()
            prompts = await client.list_prompts()

            print(f"📋 Доступно инструментов: {len(tools)}")
            for tool in tools:
                print(f"  • {tool.name}")

            print(f"\n📚 Доступно ресурсов: {len(resources)}")
            for resource in resources:
                print(f"  • {resource.uri}")

            print(f"\n💭 Доступно промптов: {len(prompts)}")
            for prompt in prompts:
                print(f"  • {prompt.name}")

            # 1. Test fetch weather of Moscow
            result = await client.call_tool("get_weather", {"city_name": "Moscow"})
            data = result.content[0].text
            print(data)

            print("\n🎉 All test completed!")
        except Exception as e:
            print(f"Error while runnig tests: {e}")


if __name__ == "__main__":
    asyncio.run(test_weather_server())
