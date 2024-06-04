import google.generativeai as genai
from PIL import Image
import cv2
import numpy as np

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

# Project Idea: History aware AI assistant

# - Periodically take screenshots of the user's screen on basis of pixel changes or time intervals
# - Analyze the screenshots and get the context of the user's screen
# - Append the analysis of all the screenshots to a text file
# - Whenever the user asks a question, the AI assistant can refer to the text file to provide a more context-aware response

# Example:
# User: "Can you remind me what I was working on last week?"
# AI: "Sure! You were working on a project related to machine learning. You were using Jupyter Notebook and running some Python code."

def analyze_screenshot(image_path):
    img = Image.open(image_path)
    # get the semantic meaning from the image
    prompt = (
        'You are the vision analysis AI that provides semantic meaning from images. '
        'Your job is to provide all the important details from the given image. '
        'Do not respond as the AI assistant to the user. '
        'Include all the necessary links, text, and data that can be extracted from the image. '
    )
    response = model.generate_content([prompt, img])
    print(response.text)
    return response.text

def compare_screenshots(image1_path, image2_path):

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    
    diff_percentage = (np.sum(diff > 50 ) / diff.size) * 100

    print(f"Percentage of difference between the two images: {diff_percentage}%")

    return diff_percentage


    
# analyze_screenshot('screenshot.jpg')
compare_screenshots('screenshot1.png', 'screenshot2.png')