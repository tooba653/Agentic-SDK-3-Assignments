import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")


provider = AsyncOpenAI(
    api_key= gemini_api_key,
    base_url= "https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model= "gemini-2.0-flash",
    openai_client= provider
)

config = RunConfig(
    model= model,
    model_provider=provider,
    tracing_disabled= True
)

language_agent = Agent(
    name="language_agent",
    instructions="Provide the official language(s) spoken in a given country or respond to any language-related question such as greetings or translations.",
    model= model
)

capital_agent = Agent(
    name="capital_agent",
    instructions="Return the capital city of a specified country.",
    model = model
)

population_agent = Agent(
    name="population_agent",
    instructions="Respond with an estimate of the current population of the specified country.",
    model= model
)


orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a country information expert. Based on the user's query, extract the country name "
        "and determine whether to fetch its capital, language, or population. Use the appropriate tool: "
        "'get_country_language', 'get_country_capital', or 'get_country_population'. "
        "Pass the country name as input to the selected tool and return the result to the user."
    ),
    model = model ,
    tools=[
        language_agent.as_tool(
            tool_name="get_country_language",
            tool_description="Get the language of the country or answer language-related questions",
   
),

        capital_agent.as_tool(
             tool_name="get_country_capital",
             tool_description="Get the capital of the country",
    
),

        population_agent.as_tool(
             tool_name="get_country_population",
             tool_description="Get the population of the country",
   
),
    ],
  
)

result = Runner.run_sync(
    orchestrator_agent,
    input="What is the capital,language and population of Turkey?",
    run_config=config
)

print(result.final_output)