import google.generativeai as genai
from PIL import Image,ImageGrab
import cv2
import numpy as np
import os
import time

generation_config = {
    'temperature': 0.5,
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
    # print(response.text)
    return response.text

def compare_screenshots(image1, image2,isImagePath=True):

    if isImagePath:
        image1 = cv2.imread(image1)
        image2 = cv2.imread(image2)
    else:
        image1 = np.array(image1)
        image2 = np.array(image2)

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    
    diff_percentage = (np.sum(diff > 50 ) / diff.size) * 100

    print(f"Percentage of difference between the two images: {diff_percentage}%")

    return diff_percentage

def take_screenshot():
    try:
        screenshot = ImageGrab.grab()
        rgb_screenshot = screenshot.convert("RGB")
        return rgb_screenshot
    except Exception as e:
        return f"An error occurred: {e}"
    
def save_screenshot(screenshot, folder='images'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{folder}/screenshot_{timestamp}.png"
    screenshot.save(filename)
    return filename

def format_analysis_result(result):
    # Replace newline characters and multiple spaces with a single space
    return ' '.join(result.split())


if __name__ == '__main__':
    try:
        prev_screenshot = take_screenshot()
        save_screenshot(prev_screenshot, folder='baseline')
        print("Baseline screenshot saved.")
        while True:
            time.sleep(5)
            curr_screenshot = take_screenshot()
            diff_percentage = compare_screenshots(prev_screenshot, curr_screenshot,isImagePath=False)
            print(f"Difference percentage: {diff_percentage}%")
            if diff_percentage > 5:
                curr_screenshot_filename = save_screenshot(curr_screenshot)
                prev_screenshot = curr_screenshot
                print("Screenshot saved.")
                analysis_result = analyze_screenshot(curr_screenshot_filename)
                formatted_result = format_analysis_result(analysis_result)
                print(f"Analysis result: {formatted_result}")

                with open('analysis_results.txt', 'a') as f:
                    f.write(f"{formatted_result}\n")

    except KeyboardInterrupt:
        print("Process interrupted by the user.")