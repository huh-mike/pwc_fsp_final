from GPTServices import ChatHistory, gpt_stream_responses
from RAGServices import fetch_relevant_documents


# Helper Functions
def update_callback(content):
    print(content, end="", flush=True)


def finished_callback(full_response):
    print(f"\n\n[End of Response]")

# Core Functions

def run_chatbotgui():
    print("Tax Assistant Chatbot. Type 'exit' to stop.")
    print("This chatbot uses RAG to find relevant tax information for your queries.")

    # Initialise Conversation
    chat_history = ChatHistory()
    # Add system message to guide the assistant's behavior
    chat_history.history = [
        {"role": "system", "content": "You are a tax assistant AI that specializes in Singapore tax information. "
                                      "Provide concise, accurate answers based on the reference information provided. "
                                      "If you don't have reference attached to answer a question, acknowledge this "
                                      "limitation and let user know that there's no relevant context and do not give advices."
                                      "There's no need to mention 'information is up to October 2023'"}
    ]

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Exiting chat test.")
            break

        # Get relevant documents using RAG
        print("Searching for relevant tax information...")
        system_rag_context = fetch_relevant_documents(user_input)

        # Create a temporary conversation for this exchange
        conversation = chat_history.get_conversation().copy()
        # Add the RAG context as a system message
        conversation.append({"role": "system", "content": f"Your reference for this question: {system_rag_context}"})
        # Add the user's question
        conversation.append({"role": "user", "content": user_input})

        # Add the user message to the history
        chat_history.add_user_message(user_input)

        print("\nAI Tax ChatBot:", end=" ", flush=True)
        # Use the temporary conversation that includes the RAG context
        gpt_stream_responses(conversation, update_callback, finished_callback, chat_history)