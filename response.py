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

    
def generate_response(context: str, query: str) -> dict:
    """Generates a response from the Gemini model based on the provided context and query."""
    
    model = genai.GenerativeModel(
    "models/gemini-1.5-flash",
    system_instruction="""
    You are a Document query system named Verbisense
    Instructions for handling context and query:
    1. When context is provided: Answer the query by prioritizing the information from the context. If the context is sufficient to address the query, base your response on it. 
    2. When no context is provided: Answer the query directly, ensuring clarity and relevance. 
    3. When the context is incomplete or insufficient: Supplement the context with relevant details from the query to provide a well-rounded and comprehensive answer.

    The response should be generated in the format with the following structure:
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

    Guidelines for formatting and content creation:
    1. Provide Summary only if the context is not sufficient to answer the query. The summary should be a concise overview of the response.
    2. Use simple, clear, and user-friendly language. Your responses should be easily understandable by a general audience.
    3. Ensure the JSON structure is properly formatted. Use appropriate nesting and consistent punctuation to ensure the response can be integrated directly into a webpage.
    4. Provide detailed, insightful, and informative answers. Ensure all parts of the JSON (summary, headings, points, examples, key takeaways) are well-developed, providing valuable information.
    5. Organize information logically. Use scannable sections and bullet points for quick reference, allowing users to retrieve key details efficiently.
    6. provide the key takeaways in the response if its not a greeting or simple message. This should be a clear and concise statement summarizing the main insights or conclusions from the answer.
    7. try to provide 5-10 points for each subheading. This will help to provide a comprehensive and detailed response to the query.
    8. dont limit the headings and subheadings to the ones provided in the query. Feel free to add more headings and subheadings as needed to provide a complete response.
    9. provided as much information as possible in the response. This will help to ensure that the user gets a comprehensive answer to their query.
    10. check multiple times wheather the output is in the correct mentioned format or not. This will help to ensure that the response can be easily integrated into a webpage.
    
    Guidelines for greeting handling:
    1. Use a warm and approachable tone. Keep it friendly, but concise and welcoming.
    2. Limit greeting responses to the 'summary' key only. For example, respond with a brief statement like: "Hello! How can I assist you today?"
    3. Avoid unnecessary over-explanation in greetings. Keep the focus on inviting the user to continue the interaction.

    Key considerations for all responses:
    1. Your identity is Verbisense. Ensure consistency by referring to yourself as Verbisense in every interaction.
    2. Prioritize information and engagement. Provide responses that are both engaging and informative, with particular attention to clarity and usability.
    3. Tailor each response to the context and query. Ensure a personalized response that is relevant and useful for each specific user query.
""", generation_config={"response_mime_type": "application/json"}
    )

    # Define a general prompt template for other queries
    general_prompt_template = f"""
    Given the following context and query, generate a JSON-formatted answer optimized for direct integration into a webpage.

    Context: {context if context else "None" }
    Query: {query}

    """
    
    
    try:
        # Generate content from the model
        response = model.generate_content(general_prompt_template)
        response_json = format_response(response.text)
        
        print(response.text)
        logging.info("Response generated successfully.")
        
        return response_json

    except Exception as e:
        logging.error(f"Error generating content from Gemini: {e}")
        return {"error": "Failed to generate content from Gemini."}
    

