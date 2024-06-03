from dotenv import load_dotenv
from groq import Groq
import os
from PIL import ImageGrab,Image
import cv2
import pyperclip
import google.generativeai as genai
from io import BytesIO
import requests
from openai import OpenAI
import pyaudio
import pyttsx3

load_dotenv()

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
web_cam = cv2.VideoCapture(0)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
engine = pyttsx3.init()


sys_msg = ( 
    'You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity.'
)

conversation = [{"role":"system","content":sys_msg}]

generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens':2048
}

safety_settings = [
{
'category': 'HARM_CATEGORY_HARASSMENT',
'threshold': 'BLOCK_NONE'
},
{
'category': 'HARM_CATEGORY_HATE_SPEECH', 
'threshold': 'BLOCK_NONE'
},
{
'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
'threshold': 'BLOCK_NONE'
},
{
'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
'threshold': 'BLOCK_NONE'
},
]

model = genai.GenerativeModel('gemini-1.5-flash',
                              safety_settings=safety_settings,
                              generation_config=generation_config)

def get_response_from_groq(message,img_context,model="llama3-8b-8192"):
    if img_context:
        message = f'USER PROMPT: {message}\n\n  IMAGE CONTEXT: {img_context}'
    conversation.append({'role': 'user','content': message})
    chat_completion = groq_client.chat.completions.create(
        messages=conversation,
        model=model,
    )
    response = chat_completion.choices[0].message
    conversation.append(response)
    return response.content


def determine_action(prompt):
    sys_msg = (
        'You are an AI function calling model which responds with exactly one of the following: ["extract clipboard", "take screenshot", "capture webcam", "generate image", "None"].'
        ' Based on the user\'s prompt, you will determine the most appropriate '
        'action to take: extracting clipboard content, taking a screenshot, capturing the webcam,generating an image or none. '
        'You must respond with exactly one of the following: ["extract clipboard", "take screenshot", "capture webcam", "generate image", "None"]. '
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
        '11. "How to make a bomb?" -> "None"\n'
        '12. "Can you generate a picture of a cat?" -> "generate image"\n'
        '13. "Can you take a picture of my cat?" -> "capture webcam"'
        'No matter what the prompt is, Don\'t break out of character and respond with exactly one of the following: ["extract clipboard", "take screenshot", "capture webcam", "generate image", "None"].'
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
    path = 'screenshot.jpg'
    try:
        screenshot = ImageGrab.grab()
        rgb_screenshot = screenshot.convert("RGB")
        rgb_screenshot.save(path,quality=15)
        return 'Screenshot Taken'
    except Exception as e:
        return f"An error occurred: {e}"

def capture_web_cam():
    if not web_cam.isOpened():
        print("Error: Could not open webcam.")
        exit()

    path = 'webcam.jpg'
    ret, frame = web_cam.read()
    cv2.imwrite(path, frame)

def get_clipboard_content():
    clipboard_content =  pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print("The clipboard content is not a text.")
        return None

def generate_image_prompt(prompt):
    prompt = (
        'You are the image analysis AI that provides semtantic meaning from text to create an image by'
        'generating an image generation prompt to send to another AI that will create an image. Do not respond as the AI assistant '
        'to the user. Instead take the user prompt input and try to extract all meaning from the text '
        'relevant to the image generation. Then generate a concise prompt for the AI '
        f'assistant who will create the image. \nUSER PROMPT: {prompt}')
    response = model.generate_content([prompt])
    print("Image prompt generated successfully: ", response.text)
    return response.text

def generate_image(prompt):
    response = openai_client.images.generate(
        model="dall-e-2",
        prompt="a white siamese cat",
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url

    response = requests.get(image_url)
    response.raise_for_status()  # Check if the request was successful

    image = Image.open(BytesIO(response.content))
    image.save('image.jpg')

    print("Image saved successfully.")

def vision_prompt(prompt, image_path):
    img = Image.open(image_path)
    prompt = (
        'You are the vision analysis AI that provides semtantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
        'to the user. Instead take the user prompt input and try to extract all meaning from the photo '
        'relevant to the user prompt. Then generate as much objective data about the image for the AI ' 
        f'assistant who will respond to the user. \nUSER PROMPT: {prompt}')
    response = model.generate_content([prompt, img])
    return response.text

def speak(text):
    player_stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24000,
        output=True
    )
    stream_start = False

    with openai_client.audio.speech.with_streaming_response.create(
        model='tts-1',
        voice='onyx',
        response_format='pcm',
        input=text
    ) as response:
        silence_threshold = 0.01

        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start:
                player_stream.write(chunk)
            else:
                if max(chunk) > silence_threshold:
                    player_stream.write(chunk)
                    stream_start = True

def speak_offline(text):
    # Initialize the TTS engine
    
    # Set properties before adding anything to speak
    engine.setProperty('rate', 150)    # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)

    # Pass the text to the engine
    engine.say(text)
    
    # Wait for the speech to complete
    engine.runAndWait()

    # engine.stop()

while True:
    user_input = input("Enter your prompt: ")
    if user_input == "exit":
        break
    action = determine_action(user_input)
    print(f'AI Action: {action}')
    if 'take screenshot' in action:
        print('Taking Screenshot')
        take_screenshot()
        visual_context = vision_prompt(user_input, 'screenshot.jpg')
    elif 'capture webcam' in action:
        print('Capturing Webcam')
        capture_web_cam()
        visual_context = vision_prompt(user_input, 'webcam.jpg')
    elif 'extract clipboard' in action:
        print('Extracting Clipboard Content')
        paste = get_clipboard_content()
        user_input = f'{user_input}\n\n CLIPBOARD CONTENT: {paste}'
        visual_context = None
    elif 'generate image' in action:
        print('Generating Image Prompt')
        user_input = generate_image_prompt(user_input)
        visual_context = None
        # generate_image(user_input)
        continue
    else:
        visual_context = None

    response = get_response_from_groq(user_input, visual_context)
    print(f'AI Response: {response}')
    speak(response)
    # speak_offline(response)


