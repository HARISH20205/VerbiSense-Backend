import os
import logging
import google.generativeai as genai
import json
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Set up logging
logging.basicConfig(level=logging.INFO)


def format_response(json_string):
    # Remove the "```json" at the start and "```" at the end
    clean_string = json_string.strip().replace("```json", "").replace("```", "").strip()
    # Convert the cleaned string to a Python dictionary
    return json.loads(clean_string)

    
def generate_response(context: str, query: str, noData: bool) -> dict:
    """Generates a response from the Gemini model based on the provided context and query."""
    
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Define a general prompt template for other queries
    general_prompt_template = f"""
    Based on the context provided below, please answer the following question in a JSON format, optimized for direct integration into a webpage:

        Context: {context}

        Question: {query}

        Your JSON response should contain each component into respective keys and values
        {{
            "summary": "A clear and concise summary of the answer.",
            "heading1": "Main Heading",
            "heading2": [
                "Subheading 1",
                "Subheading 2"
            ]
            "points": [
                "Subheading 1" : ["point 1", "point 2", ....],
                "Subheading 2" : ["point 1", "point 2", ....],
            ],
            "example": [
                "Example for Subheading 1",
                "Example for Subheading 2"
            ],
            "key_takeaways": "Key takeaways or insights from the answer."
        }}
        
        Please ensure:
        1. The summary is at the beginning.
        2. Use bullet points or numbered lists within the 'points' key.
        3. Keep the language user-friendly and accessible to a general audience.
        4. If the answer involves steps or a process, number them clearly.
        5. Maintain a clean structure for easy scanning and quick information retrieval.
        6. give detailed answer if you want create more in correct template as mentioned
    """


    print(general_prompt_template)
    try:
        # Generate content from the model
        response = model.generate_content(general_prompt_template)
        response_json = format_response(response.text)
        
        logging.info("Response generated successfully.")
        
        return response_json

    except Exception as e:
        logging.error(f"Error generating content from Gemini: {e}")
        return {"error": "Failed to generate content from Gemini."}
