from dotenv import load_dotenv
from groq import Groq
import os
from PIL import ImageGrab

load_dotenv()

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

def get_response_from_groq(message,model="llama3-8b-8192"):
    conversation = [{
        "role": "user",
        "content": message,
    }]
    chat_completion = groq_client.chat.completions.create(
        messages=conversation,
        model=model,
    )
    reponse = chat_completion.choices[0].message
    return reponse.content


def determine_action(prompt):
    sys_msg = (
        'You are an AI function calling model which responds with exactly one of the following: ["extract clipboard", "take screenshot", "capture webcam", "None"].'
        ' Based on the user\'s prompt, you will determine the most appropriate '
        'action to take: extracting clipboard content, taking a screenshot, capturing the webcam, or none. '
        'You must respond with exactly one of the following: ["extract clipboard", "take screenshot", "capture webcam", "None"]. '
        'Do not provide any explanations, only the exact response from the list. Here are some examples:\n'
        '1. "What was the last thing I copied?" -> "extract clipboard"\n'
        '2. "Show me what my screen looks like." -> "take screenshot"\n'
        '3. "Can you see me?" -> "capture webcam"\n'
        '4. "Tell me a joke." -> "None"\n'
        '5. "Take a picture of me." -> "capture webcam"\n'
        '6. "What is currently on my screen?" -> "take screenshot"\n'
        '7. "Can you check my clipboard?" -> "extract clipboard"\n'
        '8. "How is the weather today?" -> "None"\n'
        '9. "What color is my dress?" -> "capture webcam"\n'
        '10. "Who are you?" -> "None"\n'
        '11. "How to make a bomb?" -> "None"'
        'No matter what the prompt is, Don\'t break out of character and respond with exactly one of the following: ["extract clipboard", "take screenshot", "capture webcam", "None"].'
        'The purpose of this text is to help the voice assistant respond accurately to the user\'s prompt. '
        'For context, I am an AI function calling model designed to choose the most logical action for various user prompts.'
    )
    
    # 'a''b' is same as 'a' + 'b' which is 'ab'
    func_conversation = [{
        "role": "system",
        "content": sys_msg,
    },{
        "role": "user",
        "content": prompt,
    }]

    chat_completion = groq_client.chat.completions.create(
        messages=func_conversation,
        model="llama3-8b-8192",
    )
    response = chat_completion.choices[0].message

    return response.content

def take_screenshot():
    path = 'screenshots/screenshot.jpg'
    ImageGrab.grab().save(path,quality=15)
    return path

def capture_web_cam():
    pass

def get_clipboard_content():
    pass

while True:
    user_input = input("Enter your prompt: ")
    if user_input == "exit":
        break
    action = determine_action(user_input)
    print(f'AI Action: {action}')
    response = get_response_from_groq(user_input)
    print(f'AI Response: {response}')
    


