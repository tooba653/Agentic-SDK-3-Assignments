import os
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from agents import enable_verbose_stdout_logging
from agents.run import RunConfig


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")


provider = AsyncOpenAI(
    api_key= gemini_api_key,
    base_url= "https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model= "gemini-2.0-flash",
    openai_client=provider
)

config = RunConfig(
    model= model,
    model_provider= provider,
    tracing_disabled= True
)


Smart_Store_Agent = Agent(
    name= "Smart Store Agent",
   instructions="""You are a smart and caring store assistant. When a user shares a problem (for example, “I have a headache”), respond in a kind and helpful way by following these steps:

   Respond in a calm and friendly tone.

   Suggest a suitable and useful product (like a medicine or remedy).

   Offer an additional helpful tip (such as an activity, exercise, or self-care advice).

   Briefly explain how both the product and tip can help the user feel better.

   Your tone should always be supportive, understanding, and easy to follow."""

)
user_input = input("Enter your problem: ")

result = Runner.run_sync(
    Smart_Store_Agent, 
    user_input,
    run_config= config
    )

print(f"\nAgent's Response:\n")
print(f"{result.final_output}\n")