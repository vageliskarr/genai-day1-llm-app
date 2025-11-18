import os
from dotenv import load_dotenv
from openai import OpenAI

# Φορτώνουμε τις μεταβλητές από το .env
load_dotenv()

# Παίρνουμε το API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in .env file")

# Δημιουργούμε OpenAI client
client = OpenAI(api_key=api_key)

def ask_llm(prompt: str) -> str:
    """
    Στέλνει ένα prompt στο LLM και επιστρέφει την απάντηση.
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",   # οικονομικό και γρήγορο μοντέλο
        messages=[
            {
                "role": "system",
                "content": "You are a concise, helpful AI assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content
