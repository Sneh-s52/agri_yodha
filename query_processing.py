# --- 1. CONFIGURATION ---
import os
import io
from dotenv import load_dotenv
import vertexai # <-- ADDED
from vertexai.generative_models import GenerativeModel, Part # <-- ADDED
from google.cloud import translate_v2 as translate
from google.cloud import speech
from PIL import Image
import cv2
from moviepy import VideoFileClip
from google.oauth2 import service_account

# Load environment variables from .env file
load_dotenv()

SERVICE_ACCOUNT_FILE = "C:/Users/adwik/Downloads/focal-slice-452619-a9-0466f27bceb9.json"

# Instantiate Google Cloud clients
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
translate_client = translate.Client(credentials=credentials)
speech_client = speech.SpeechClient(credentials=credentials)

# --- ADD VERTEX AI INITIALIZATION ---
# ❗️ IMPORTANT: Replace with your actual GCP Project ID and a supported region
# The Project ID should be the one associated with your service account file.
PROJECT_ID = "focal-slice-452619-a9" 
REGION = "us-east4" # Example region, change if needed
vertexai.init(project=PROJECT_ID, location=REGION, credentials=credentials)


# --- 2. CORE PROCESSING FUNCTIONS ---

def detect_language(text_query):
    """Detects the language of the input text."""
    result = translate_client.detect_language(text_query)
    print(f"✅ Language Detected: {result['language']} (Confidence: {result['confidence']:.2f})")
    return result['language']

def process_image_input(image_path, text_query):
    """Processes image + text input using the Vertex AI Gemini 1.0 Pro Vision model."""
    print("🧠 Processing Image + Text Input (Vertex AI)...")
    try:
        # Load the image and convert to bytes
        with Image.open(image_path) as img:
            img_byte_arr = io.BytesIO()
            # Preserve format (e.g., JPEG, PNG) to create the correct mime type
            img_format = img.format if img.format else 'JPEG'
            img.save(img_byte_arr, format=img_format)
            image_bytes = img_byte_arr.getvalue()

        # Create a Part object for the image
        image_part = Part.from_data(
            data=image_bytes, mime_type=f"image/{img_format.lower()}"
        )

        # Instantiate the model from Vertex AI
        model = GenerativeModel("gemini-1.5-pro-001")
        
        prompt_parts = [
            image_part,
            "Analyze the attached image in detail and then translate the following user query into English. Combine the analysis and the translated query into a single, cohesive English prompt for a backend system.",
            f"User Query: '{text_query}'",
        ]
        
        response = model.generate_content(prompt_parts)
        print("✅ Vertex AI Vision processing complete.")
        return response.text
    except FileNotFoundError:
        return f"Error: Image file not found at {image_path}"
    except Exception as e:
        return f"An error occurred during image processing: {e}"

def process_audio_input(audio_path, source_language_code=None):
    """Processes audio input by transcribing and then translating."""
    print("🧠 Processing Audio Input...")
    try:
        with io.open(audio_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=source_language_code if source_language_code else "en-US",
            enable_automatic_punctuation=True
        )

        print("   - Transcribing audio...")
        response = speech_client.recognize(config=config, audio=audio)
        if not response.results:
            return "Error: Could not transcribe audio."
        
        transcription = response.results[0].alternatives[0].transcript
        print(f"   - Transcription successful: '{transcription}'")

        print("   - Translating transcription to English...")
        translation_result = translate_client.translate(transcription, target_language='en')
        english_prompt = translation_result['translatedText']
        print("✅ Audio processing complete.")
        return english_prompt

    except FileNotFoundError:
        return f"Error: Audio file not found at {audio_path}"
    except Exception as e:
        return f"An error occurred during audio processing: {e}"

def process_text_input(text_query):
    """Processes text-only input by translating it to English."""
    print("🧠 Processing Text Input...")
    result = translate_client.translate(text_query, target_language='en')
    print("✅ Text processing complete.")
    return result['translatedText']

def process_video_input(video_path, text_query):
    """Processes video input by extracting audio and key frames."""
    print("🎬 Processing Video Input (Vertex AI)...")
    temp_audio_path = "temp_audio.wav"
    
    try:
        # --- Part 1 and 2: Audio and Frame Extraction (No changes needed here) ---
        print("   - Extracting audio from video...")
        with VideoFileClip(video_path) as video:
            video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le', ffmpeg_params=["-ar", "16000"])
        audio_prompt = process_audio_input(temp_audio_path)
        os.remove(temp_audio_path)

        print("   - Extracting key frames from video...")
        video_capture = cv2.VideoCapture(video_path)
        frames = []
        frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
        frame_interval = int(frame_rate * 5)
        frame_count = 0
        while True:
            success, frame = video_capture.read()
            if not success:
                break
            if frame_count % frame_interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                frames.append(pil_img)
            frame_count += 1
        video_capture.release()
        
        if not frames:
            print("   - Could not extract frames, proceeding with audio-only analysis.")
            return audio_prompt

        print(f"   - Extracted {len(frames)} frames for analysis.")

        # --- Part 3: Analyze Frames with Vertex AI Gemini Vision (UPDATED SECTION) ---
        print("   - Analyzing frames with Vertex AI Vision...")
        model = GenerativeModel("gemini-1.5-pro-002") # <-- Use Vertex AI Model
        
        prompt_parts = [
            f"Analyze these video frames in sequence. The audio from the video has already been transcribed and translated to: '{audio_prompt}'.",
            "Based on both the audio and the visual frames, analyze the full context and then address the following user query.",
            f"User Query: '{text_query}'",
        ]
        # Convert PIL images to Part objects for Vertex AI
        for pil_img in frames:
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='JPEG')
            image_part = Part.from_data(data=img_byte_arr.getvalue(), mime_type="image/jpeg")
            prompt_parts.append(image_part)

        response = model.generate_content(prompt_parts) # <-- Same generate_content call
        
        print("✅ Video processing complete.")
        return response.text

    except FileNotFoundError:
        return f"Error: Video file not found at {video_path}"
    except Exception as e:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return f"An error occurred during video processing: {e}"

def run_backend_logic(english_prompt):
    """--- PLACEHOLDER for your Agents/Orchestrator Framework ---"""
    print("🚀 Sending to Backend Agents/Orchestrator...")
    print(f"   - Input to backend: '{english_prompt}'")
    backend_answer = f"This is the final answer from the reasoning LLM based on the prompt. The core request seems to be about '{english_prompt[:50]}...'"
    print("✅ Backend processing complete.")
    return backend_answer

def translate_to_original_language(text_to_translate, target_language):
    """Translates the final English answer back to the original language."""
    print(f"🌐 Translating final answer to '{target_language}'...")
    result = translate_client.translate(text_to_translate, target_language=target_language)
    print("✅ Final translation complete.")
    return result['translatedText']


# --- 3. MAIN WORKFLOW ORCHESTRATOR ---

def main_workflow(input_data):
    """Orchestrates the entire flowchart from input to final output."""
    input_type = input_data.get('type')
    english_prompt = ""
    original_language = "en"

    if input_type == 'image':
        query = input_data.get('query', '')
        path = input_data.get('path')
        original_language = detect_language(query) if query else 'en'
        english_prompt = process_image_input(path, query)
    
    elif input_type == 'audio':
        path = input_data.get('path')
        original_language = input_data.get('query_lang', 'en-US').split('-')[0]
        english_prompt = process_audio_input(path, source_language_code=input_data.get('query_lang'))

    elif input_type == 'video':
        query = input_data.get('query', '')
        path = input_data.get('path')
        original_language = detect_language(query) if query else 'en'
        english_prompt = process_video_input(path, query)

    elif input_type == 'text':
        query = input_data.get('query', '')
        original_language = detect_language(query)
        english_prompt = process_text_input(query)

    else:
        return "Error: Invalid input type specified."

    final_answer_english = run_backend_logic(english_prompt)
    final_answer_translated = translate_to_original_language(final_answer_english, original_language)

    return final_answer_translated


# --- 4. EXAMPLE USAGE ---

if __name__ == "__main__":
    # ⚠️ IMPORTANT: Create dummy files for these examples to work.
    # 1. Create an image file named `image1.jpg`.
    # 2. Create a WAV audio file named `sample_audio.wav`.
    # 3. Create a short video file named `sample_video.mp4`.

    # --- CHOOSE ONE EXAMPLE TO RUN ---

    # # Example 1: Text Input (Tamil)
    # text_input = {
    #     'type': 'text',
    #     'query': 'இன்றைய வானிலை என்ன?' # "What is the weather today?"
    # }
    # final_result = main_workflow(text_input)

    # Example 2: Image + Text Input (Hindi)
    image_input = {
        'type': 'image',
        'query': 'यह किस फसल का पौधा है?', # "What crop plant is this?"
        'path': 'image1.jpg' 
    }
    final_result = main_workflow(image_input)

    # # Example 3: Audio Input (Hindi)
    # audio_input = {
    #     'type': 'audio',
    #     'query_lang': 'hi-IN',
    #     'path': 'sample_audio.wav'
    # }
    # final_result = main_workflow(audio_input)

    # # Example 4: Video + Text Input (Hindi)
    # video_input = {
    #    'type': 'video',
    #    'query': 'इस वीडियो में क्या हो रहा है? संक्षेप में बताएं।', # "What is happening in this video? Explain in brief."
    #    'path': 'sample_video.mp4'
    # }
    # final_result = main_workflow(video_input)


    print("\n" + "="*50)
    print("🎉 FINAL OUTPUT TO USER 🎉")
    print(final_result)
    print("="*50)