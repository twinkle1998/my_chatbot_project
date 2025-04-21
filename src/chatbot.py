class ChatBot:
    def __init__(self):
        self.responses = {
            "hello": "Hi there!",
            "how are you": "I'm doing great, thanks for asking!",
            "bye": "Goodbye! See you soon!"
        }

    def get_response(self, message: str) -> str:
        message = message.lower().strip()
        return self.responses.get(message, "Sorry, I don't understand that. Try saying 'hello' or 'bye'!")