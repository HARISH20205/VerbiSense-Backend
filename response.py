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
    clean_string = json_string.strip().replace("```json", "").replace("```", "").replace("*","").replace("`","").strip()
    # Convert the cleaned string to a Python dictionary
    return json.loads(clean_string)

    
def generate_response(context: str, query: str, noData: bool) -> dict:
    """Generates a response from the Gemini model based on the provided context and query."""
    
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Define a general prompt template for other queries
    general_prompt_template = f"""
    Given the following context and query, generate a JSON-formatted answer optimized for direct integration into a webpage.

    Context: {context if context else "None" }
    Query: {query}

    If the context is provided, answer the query based on the context. If the context does not fully address the query, generate the answer by expanding on the query. If no context is provided, generate the answer solely based on the query.

    Your JSON response should follow this structure:
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

    Guidelines for query handling:
    1. **When context is provided**: Answer the query based on the context. If the context is sufficient, align your response with it.
    2. **When context is not provided**: Generate the answer solely based on the query, ensuring it's relevant and coherent.
    3. **If the context does not fully answer the query**: Expand the response to provide a relevant answer, filling in any gaps from the query itself.

    Guidelines for greetings:
    1. Use a friendly and approachable tone.
    2. Keep the response brief and focused on engagement, with no need for over-explanation.
    3. Offer a friendly invitation to continue the interaction, such as 'How can I assist you today?'
    4. For greetings, limit the output to the 'summary' key only in the JSON format.

    Key considerations:
    1. Start every response with the summary.
    2. Use accessible and user-friendly language.
    3. Ensure the JSON structure is clean and correct, with proper nesting and formatting.
    4. For greetings, keep responses concise, while for other queries, provide detailed and informative answers.
    5. Structure responses to be easily scannable for quick information retrieval.
    6. Your name is Verbisense; maintain this identity consistently and do not refer to any other names or personas.
    """

    try:
        # Generate content from the model
        response = model.generate_content(general_prompt_template)
        response_json = format_response(response.text)
        
        logging.info("Response generated successfully.")
        
        return response_json

    except Exception as e:
        logging.error(f"Error generating content from Gemini: {e}")
        return {"error": "Failed to generate content from Gemini."}
