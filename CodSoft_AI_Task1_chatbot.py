def chatbot():
    """
    Function to run a simple rule-based chatbot.
    """
    print("Hi! I'm a simple rule-based chatbot. How can I help you today?")
    
    while True:
        # Take user input and convert it to lowercase for case-insensitive matching
        user_input = input("You: ").lower()
        
        # Greeting response
        if "hello" in user_input or "hi" in user_input:
            print("Chatbot: Hello! How can I assist you today?")
        # Respond to inquiries about the chatbot's well-being
        elif "how are you" in user_input:
            print("Chatbot: I'm just a program, but I'm here to help you!")
        # Respond to inquiries about the chatbot's name
        elif "your name" in user_input:
            print("Chatbot: I'm a simple chatbot created to assist you.")
        # Handle user wanting to exit the conversation
        elif "bye" in user_input or "exit" in user_input:
            print("Chatbot: Goodbye! Have a great day!")
            break
        # Respond to requests for help
        elif "help" in user_input:
            print("Chatbot: Sure, I can help you with basic queries. What do you need help with?")
        # Default response for unrecognized input
        else:
            print("Chatbot: I'm sorry, I didn't understand that. Can you please rephrase?")

if __name__ == "__main__":
    # Run the chatbot function
    chatbot()
