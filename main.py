import streamlit as st
from PIL import Image
import easyocr
from groq import Groq
import numpy as np

# Set up the page
st.set_page_config(page_title="OCR + LLM Text Analyzer", page_icon="✍️")
st.title("Image to Text Analysis with OCR and LLM")
st.markdown("---")

# Initialize EasyOCR reader once to save resources
@st.cache_resource
def get_easyocr_reader():
    """Caches the EasyOCR reader to avoid reloading it on every run."""
    return easyocr.Reader(['es', 'en']) # You can add more languages here

reader = get_easyocr_reader()

# Groq API Key Input
with st.expander("🔑 **Enter your Groq API Key**"):
    groq_api_key = st.text_input("Groq API Key", type="password")
    st.info("You can get a Groq API Key from [https://groq.com/](https://groq.com/)")

# Image Uploader
st.header("1. Upload your image")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.markdown("---")

    # OCR Processing with EasyOCR
    st.header("2. Extracting text with OCR")
    with st.spinner("Extracting text..."):
        try:
            # Convert PIL Image to a NumPy array for EasyOCR
            image_np = np.array(image)
            
            # Read text from the image
            results = reader.readtext(image_np)
            
            # Concatenate all detected text into a single string
            extracted_text = " ".join([text for (bbox, text, prob) in results])
            
            if extracted_text.strip():
                st.success("Text extracted successfully!")
                st.text_area("Extracted Text", extracted_text, height=250)
                st.markdown("---")
            else:
                st.warning("No text was detected in the image.")
                extracted_text = None
        except Exception as e:
            st.error(f"Error during OCR: {e}")
            extracted_text = None

    # LLM Analysis
    if extracted_text and groq_api_key:
        st.header("3. Analyzing text with LLM")
        
        if st.button("Generate Analysis"):
            if not groq_api_key.strip():
                st.error("Please enter your Groq API key to proceed.")
            else:
                with st.spinner("Generating analysis, summary, and reflection..."):
                    try:
                        client = Groq(api_key=groq_api_key)
                        
                        # Define the prompt for the LLM
                        prompt = (
                            f"Analyze the following text from an image. Provide a detailed analysis, a concise summary, and a thoughtful reflection on the content.\n\n"
                            f"**Text:**\n{extracted_text}\n\n"
                            f"**Instructions:**\n"
                            f"1. **Analysis:** Break down the key points, themes, and overall purpose of the text.\n"
                            f"2. **Summary:** Provide a brief, one-paragraph summary of the content.\n"
                            f"3. **Reflection:** Offer a personal or general reflection on the meaning or implications of the text.\n"
                            f"**Format the response clearly with these three headings.**"
                        )
                        
                        chat_completion = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt,
                                }
                            ],
                            model="llama3-8b-8192", # You can choose a different model if desired
                        )
                        
                        llm_response = chat_completion.choices[0].message.content
                        
                        # Display the LLM's response
                        st.subheader("LLM Results")
                        st.write(llm_response)
                        
                    except Exception as e:
                        st.error(f"Error communicating with the Groq API: {e}")
