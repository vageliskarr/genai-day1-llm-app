from llm_client import ask_llm

def main():
    print("Simple LLM console app. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Bye!")
            break

        try:
            answer = ask_llm(user_input)
            print(f"\nAI: {answer}\n")
        except Exception as e:
            print(f"Error calling LLM: {e}")

if __name__ == "__main__":
    main()
