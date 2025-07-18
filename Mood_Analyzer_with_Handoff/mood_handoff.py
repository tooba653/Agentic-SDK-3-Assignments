import os
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from agents.run import RunConfig

load_dotenv()


gemini_api_key = os.getenv("GEMINI_API_KEY")

provider= AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)


config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

Mood_Suggestions = Agent(
    name="Mood Suggestions",
   instructions="""Based on the user's emotional state, your role is to offer kind and helpful suggestions.If the user appears 'sad' or 'stressed', recommend an uplifting activity, a relaxing method, or useful advice that can help improve their mood.Keep your tone gentle, motivating, and caring.""",
)


mood_analyst = Agent(
    name="Mood Analyst",
    instructions="You detect the user's mood and pass it to the suitable agent.",
    handoffs=[Mood_Suggestions],
)


user_input = input("Enter your mood: ")

result_suggestions = Runner.run_sync(
    Mood_Suggestions,
    user_input,
    run_config=config
)

result_analyst = Runner.run_sync(
    mood_analyst,
    user_input,
    run_config=config
)

output_content = f"""
Agent Responses:

Mood Suggestions Agent Response:
{result_suggestions.final_output}
Responded by: {Mood_Suggestions.name}

Mood Analyst Agent Response:
{result_analyst.final_output}
Responded by: {result_analyst.last_agent.name}
"""

output_file = "agent_responses.txt"
with open(output_file, "w") as file:
    file.write(output_content)

print(f"Responses have been written to {output_file}")


print(output_content)