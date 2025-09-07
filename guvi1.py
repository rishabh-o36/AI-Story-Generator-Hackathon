# Set environment variable to potentially help with CUDA memory fragmentation
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
print("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")


import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline
import torch
import re
import io
import zipfile
from PIL import Image
# import reportlab # Uncomment if implementing PDF export

st.set_page_config(page_title="AI Story Generator with Image", page_icon="")
st.title("📙 AI STORY GENERATOR WITH IMAGE")
st.markdown("Type a creative prompt, choose your settings, and let the AI generate a multi-part story with images!")

# --- User Input Section ---
st.subheader("Story Idea and Settings")
prompt = st.text_input("🖋 Enter the core story idea:", "A young girl finds a secret door in her grandmother's attic.")

# Optional settings - these can be used later to influence text generation
genre = st.selectbox("Select Genre:", ["Fantasy", "Sci-Fi", "Mystery", "Comedy", "Other"])
tone = st.selectbox("Select Tone:", ["Lighthearted", "Dark", "Epic", "Humorous", "Serious"])
audience = st.selectbox("Target Audience:", ["Kids", "Teens", "Adults", "All"])

length = st.slider("🖋 Max Story Length per part (tokens)", min_value=100, max_value=300, value=200, step=20) # Increased max length for multi-part story

# --- Local Model Paths ---
# IMPORTANT: Replace these paths with actual valid Hugging Face model IDs or local paths to models
# formatted correctly for AutoTokenizer.from_pretrained and StableDiffusionPipeline.from_pretrained
LOCAL_TEXT_MODEL_PATH = "gpt2" # Example: Using a public model for demonstration
# Consider using a smaller model or torch_dtype=torch.float16 if you encounter CUDA memory issues
LOCAL_IMAGE_PIPELINE_PATH = "runwayml/stable-diffusion-v1-5" # Example: Using a public model for demonstration


# --- Load Text Generation Model and Tokenizer ---
@st.cache_resource
def load_text_model(model_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        # Move to GPU if available (optional, depends on local setup)
        if torch.cuda.is_available():
            model.to("cuda")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading text model: {e}. Please ensure LOCAL_TEXT_MODEL_PATH is a valid Hugging Face model ID or a correctly formatted local path.")
        return None, None

tokenizer, text_model = load_text_model(LOCAL_TEXT_MODEL_PATH)


# --- Load Image Generation Pipeline ---
@st.cache_resource
def load_image_pipeline(pipeline_path):
    try:
        # Use torch_dtype=torch.float16 to potentially reduce memory usage
        pipe = StableDiffusionPipeline.from_pretrained(pipeline_path, use_safetensors=True, torch_dtype=torch.float16)
        # Move to GPU if available (optional, depends on local setup)
        if torch.cuda.is_available():
            pipe.to("cuda")
        else:
            st.warning("CUDA not available, using CPU. Image generation may be slow.")
        return pipe
    except Exception as e:
        st.error(f"Error loading image pipeline: {e}. Please ensure LOCAL_IMAGE_PIPELINE_PATH is a valid Hugging Face model ID or a correctly formatted local path.")
        return None

image_pipe = load_image_pipeline(LOCAL_IMAGE_PIPELINE_PATH)


# --- Function to generate image prompt based on story segment content ---
def generate_image_prompt(segment_text, part, genre, tone):
    # Simple keyword extraction (can be enhanced with more sophisticated NLP)
    keywords = re.findall(r'\b\w{4,}\b', segment_text) # Extract words with 4 or more characters
    # Filter out common words and keep unique ones
    common_words = set(["the", "and", "a", "of", "to", "in", "is", "it", "that", "on", "with", "for", "by", "this", "about", "are", "from", "was", "were"])
    filtered_keywords = [word for word in keywords if word.lower() not in common_words]
    unique_keywords = list(set(filtered_keywords))

    # Construct the prompt - make it more descriptive for better image generation and add emphasis on color
    image_prompt = f"Vibrant and detailed illustration for the {part} of a {genre} story with a {tone} tone. Scene focuses on: {', '.join(unique_keywords[:15])}. Full scene description: {segment_text[:300]}. Emphasize color and completeness." # Increased keywords and description length, added emphasis on color and completeness

    return image_prompt

# --- Generation Button ---
if st.button("Generate Story and Images"):
    if tokenizer is not None and text_model is not None and image_pipe is not None:
        with st.spinner("Generating your story and images..."):
            story_parts = ["Introduction", "Conflict", "Climax", "Resolution"]
            full_story = ""
            story_segments_for_images = []
            previous_text = "" # Keep track of previous text for coherence

            for i, part in enumerate(story_parts):
                # Refined prompt for text generation
                # We'll provide the overall prompt and the previous text to guide the generation
                if part == "Introduction":
                    part_prompt = f"Write a {genre} story with a {tone} tone for {audience} about: {prompt}\n\n{part}:"
                else:
                     # Provide more context for subsequent parts and explicitly ask for a continuation
                     part_prompt = f"{full_story.strip()}\n\nWrite the {part} of the story, continuing from the previous part:"


                inputs = tokenizer(part_prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {key: value.to("cuda") for key, value in inputs.items()}

                # Adjust max_length for subsequent parts to avoid exceeding limits
                # Also increase max_length slightly for Climax to encourage more text
                current_max_length = len(tokenizer(part_prompt).input_ids) + length
                if i > 0: # For Conflict, Climax, Resolution, adjust max length based on previous text
                    current_max_length = len(tokenizer(full_story).input_ids) + length + (50 if part == "Climax" else 0) # Add extra tokens for Climax


                generated_ids = text_model.generate(
                    **inputs,
                    max_length=current_max_length,
                    num_return_sequences=1,
                    temperature=0.85,
                    top_k=50,
                    top_p=0.92,
                    repetition_penalty=1.5,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.eos_token_id
                )
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                # Remove the input prompt from the generated text
                # Use regex to remove the part prompt more reliably
                generated_text = re.sub(re.escape(part_prompt), "", generated_text, 1).strip()


                # Update previous_text for the next iteration
                previous_text = generated_text


                full_story += generated_text + "\\n\\n" # Add a newline for separation

                # Generate a more descriptive image prompt
                image_prompt = generate_image_prompt(generated_text.strip(), part, genre, tone)
                story_segments_for_images.append({"text": generated_text.strip(), "prompt_for_image": image_prompt, "image": None}) # Add a placeholder for the generated image


            st.subheader("Generated Story")

            # --- Display Story Segments and Generate Images ---
            for i, segment in enumerate(story_segments_for_images):
                st.markdown(f"**{story_parts[i]}:**")
                st.write(segment["text"])

                # --- Image Generation ---
                if segment.get("prompt_for_image"):
                    try:
                        image = image_pipe(segment["prompt_for_image"]).images[0]
                        segment["image"] = image # Store the generated image
                        st.image(image, caption=f"{story_parts[i]} Illustration")
                    except Exception as e:
                        st.error(f"Error generating image for {story_parts[i]}: {e}")
                else:
                    st.info(f"No image prompt generated for {story_parts[i]}.")


            # --- Optional Export/Enhancements Section ---
            st.markdown("---")
            st.subheader("Options")

            # Store story data in session state for export
            st.session_state['story_data'] = story_segments_for_images
            st.session_state['full_story_text'] = full_story
    else:
        st.error("Model loading failed. Please check your model paths.")


# --- Export Functionality ---
if 'story_data' in st.session_state and st.session_state['story_data']:
    story_segments_for_images = st.session_state['story_data']
    full_story_text = st.session_state['full_story_text']
    story_parts = ["Introduction", "Conflict", "Climax", "Resolution"] # Define story_parts again for the export section

    # Export as Text File
    st.download_button(
        label="Export as Text File",
        data=full_story_text,
        file_name="story.txt",
        mime="text/plain"
    )

    # Export as Image Gallery (ZIP file of images)
    if any(segment["image"] is not None for segment in story_segments_for_images):
        with io.BytesIO() as archive:
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for i, segment in enumerate(story_segments_for_images):
                    if segment["image"]:
                        img_byte_arr = io.BytesIO()
                        segment["image"].save(img_byte_arr, format='PNG')
                        zipf.writestr(f'scene_{i+1}_{story_parts[i].lower().replace(" ", "_")}.png', img_byte_arr.getvalue())
            st.download_button(
                label="Export as Image Gallery (ZIP)",
                data=archive.getvalue(),
                file_name="story_images.zip",
                mime="application/zip"
            )

    # Placeholder for Export as PDF (requires additional libraries like reportlab)
    # st.markdown("*(PDF export requires additional libraries and implementation)*")

    # Placeholder for Export as Storybook (requires custom formatting/HTML generation)
    # st.markdown("*(Storybook export requires custom formatting and implementation)*")

# --- Team Attribution ---
st.markdown("---") # Add a separator line
st.markdown("<p style='text-align: center;'>Team: The Glitch</p>", unsafe_allow_html=True) # Centered attribution
