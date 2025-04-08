import streamlit as st
from groq import Groq
import base64
from PIL import Image as PILImage
import os
from dotenv import load_dotenv
import time
import hashlib
from typing import Optional, Tuple

# Load environment variables from .env file
load_dotenv()

# Constants
TEMP_IMAGE_DIR = "temp_images"
CACHE_DIR = "cache"
MAX_RETRIES = 3
RETRY_DELAY = 2

# Create necessary directories if they don't exist
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Get Groq API key from the environment
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()

# Initialize Groq API client with API key
try:
    client = Groq(api_key=groq_api_key)
except Exception as e:
    st.error(f"Failed to initialize Groq client: {str(e)}")
    st.stop()

# Model configurations
MODELS = {
    "LLaVA": 'llama-3.2-11b-vision-preview',
    "Llama 3.1": 'llama-3.3-70b-versatile'
}

# Language options
LANGUAGES = {
    "Marathi": {
        "description_prompt": "Describe this image in detail in Marathi, including the appearance of the dog(s) and any notable actions or behaviors.",
        "system_prompt": "You're a top-tier Marathi comedy writer — your stories are full of laughs, twists, and punchlines. Write a hilarious, unforgettable story in Marathi based on the scene in this image."
    },
    "English": {
        "description_prompt": "Describe this image in detail in English, including the appearance of the dog(s) and any notable actions or behaviors.",
        "system_prompt": "You're a top-tier comedy writer — your stories are full of laughs, twists, and punchlines. Write a hilarious, unforgettable story in English based on the scene in this image."
    },
    "Hindi": {
        "description_prompt": "Describe this image in detail in Hindi, including the appearance of the dog(s) and any notable actions or behaviors.",
        "system_prompt": "You're a top-tier Hindi comedy writer — your stories are full of laughs, twists, and punchlines. Write a hilarious, unforgettable story in Hindi based on the scene in this image."
    }
}

# Function to generate cache key
def generate_cache_key(image_path: str, operation: str, language: str) -> str:
    """Generate a unique cache key based on image content, operation, and language."""
    with open(image_path, "rb") as f:
        image_hash = hashlib.md5(f.read()).hexdigest()
    return f"{image_hash}_{operation}_{language}"

# Function to check cache
def check_cache(cache_key: str) -> Optional[str]:
    """Check if result exists in cache."""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.txt")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()
    return None

# Function to save to cache
def save_to_cache(cache_key: str, content: str):
    """Save result to cache."""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.txt")
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(content)

# Function to resize image if too large
def resize_image(image: PILImage.Image, max_size: Tuple[int, int] = (800, 800)) -> PILImage.Image:
    """Resizes the image if it's larger than max_size."""
    try:
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size)
        return image
    except Exception as e:
        st.error(f"Error resizing image: {str(e)}")
        return image

# Function to encode image to base64
def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        st.error(f"Error encoding image: {str(e)}")
        raise

# Function to generate image description using LLaVA with retries
def image_to_text(client: Groq, model: str, base64_image: str, prompt: str) -> str:
    """Generate image description with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                model=model
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_DELAY)
    raise Exception("Max retries exceeded")

# Function to generate short story using Llama 3.1 with retries
def short_story_generation(client: Groq, model: str, image_description: str, system_prompt: str) -> str:
    """Generate short story with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": image_description,
                    }
                ],
                model=model
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_DELAY)
    raise Exception("Max retries exceeded")

# Function to clean up temporary files
def cleanup_temp_files():
    """Remove all files in the temporary directory."""
    try:
        for filename in os.listdir(TEMP_IMAGE_DIR):
            file_path = os.path.join(TEMP_IMAGE_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                st.warning(f"Failed to delete {file_path}: {e}")
    except Exception as e:
        st.warning(f"Error during temp file cleanup: {e}")

# Streamlit app
def main():
    st.title("LLaVA & Llama 3.1: Image Description and Story Generator")
    
    # Language selection
    selected_language = st.selectbox("Select Language", list(LANGUAGES.keys()))
    language_config = LANGUAGES[selected_language]
    
    # Image upload section
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    if uploaded_image:
        try:
            # Open the uploaded image using PIL
            with st.spinner("Processing image..."):
                image = PILImage.open(uploaded_image)
                
                # Resize the image if it's too large
                resized_image = resize_image(image)
                
                # Save resized image to the temp_images directory
                image_path = os.path.join(TEMP_IMAGE_DIR, uploaded_image.name)
                resized_image.save(image_path)
                
                # Display the uploaded image
                st.image(resized_image, caption=f"Uploaded Image (Resized) - {selected_language}", use_column_width=True)
                
                # Encode the image to base64
                base64_image = encode_image(image_path)
                
                # Check cache for image description
                description_cache_key = generate_cache_key(image_path, "description", selected_language)
                cached_description = check_cache(description_cache_key)
                
                if cached_description:
                    st.success("Loaded description from cache!")
                    image_description = cached_description
                else:
                    # Generate image description with progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Generating image description using LLaVA...")
                    progress_bar.progress(30)
                    
                    image_description = image_to_text(
                        client, 
                        MODELS["LLaVA"], 
                        base64_image, 
                        language_config["description_prompt"]
                    )
                    
                    progress_bar.progress(70)
                    save_to_cache(description_cache_key, image_description)
                    progress_bar.progress(100)
                    status_text.text("Description generated successfully!")
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                
                st.write("### Image Description")
                st.write(image_description)
                
                # Check cache for story
                story_cache_key = generate_cache_key(image_path, "story", selected_language)
                cached_story = check_cache(story_cache_key)
                
                if cached_story:
                    st.success("Loaded story from cache!")
                    short_story = cached_story
                else:
                    # Generate short story with progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Generating short story using Llama 3.1...")
                    progress_bar.progress(30)
                    
                    short_story = short_story_generation(
                        client, 
                        MODELS["Llama 3.1"], 
                        image_description, 
                        language_config["system_prompt"]
                    )
                    
                    progress_bar.progress(70)
                    save_to_cache(story_cache_key, short_story)
                    progress_bar.progress(100)
                    status_text.text("Story generated successfully!")
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                
                st.write("### Generated Story")
                st.write(short_story)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please try again or check the console for more details.")
            raise e
        
        finally:
            # Clean up temporary files
            cleanup_temp_files()

if __name__ == "__main__":
    main()