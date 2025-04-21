from fastapi import FastAPI
import uvicorn
from chatbot import ChatBot  # Assuming this is your chatbot module

app = FastAPI()

# Root route to handle requests to "/"
@app.get("/")
async def root():
    return {"message": "Welcome to my chatbot project!"}

# Chat route to handle chatbot interactions
@app.get("/chat")
async def chat(message: str):
    chatbot = ChatBot()
    response = chatbot.get_response(message)
    return {"response": response}

# Favicon route to handle browser favicon requests
@app.get("/favicon.ico")
async def favicon():
    return {"detail": "No favicon available"}

# Run the app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
