import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from time import perf_counter as timer
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import logging
import requests
import re


# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Importing processors (assumed to be your custom modules)
from src.text_processor import process_text_file
from src.audio_processor import process_audio_file
from src.video_processor import process_video_file
from src.image_processor import process_image_file

# Hugging Face API settings
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")


def process_files(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Processes a list of files in parallel and returns their processed content."""
    
    def process_single_file(file_path):
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()
        file_name = os.path.basename(file_path)

        # Validate if file exists
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            logging.error(f"File {file_name} does not exist.")
            return []

        try:
            if extension in ['.txt', '.pdf', '.docx']:
                return process_text_file(file_path)
            elif extension in ['.mp3', '.wav', '.flac']:
                return process_audio_file(file_path)
            elif extension in ['.mp4']:
                return process_video_file(file_path)
            elif extension in ['.png', '.jpg', '.jpeg']:
                return process_image_file(file_path)
            else:
                logging.warning(f"Unsupported file type: {extension} for file {file_name}")
                return []
        except Exception as e:
            logging.error(f"Error processing file {file_name}: {e}", exc_info=True)
            return []

    # Process files in parallel, limiting threads to the number of CPU cores
    with ThreadPoolExecutor(max_workers=min(len(file_paths), os.cpu_count())) as executor:
        results = executor.map(process_single_file, file_paths)

    # Flatten the results
    processed_data = [item for result in results for item in result]

    return processed_data


def create_embeddings(processed_data: List[Dict[str, Any]], embedding_model: SentenceTransformer) -> pd.DataFrame:
    """Generates embeddings for processed data."""
    try:
        text_chunks = [item["text"] for item in processed_data]
        embeddings = torch.empty((0, embedding_model.get_sentence_embedding_dimension()), device=embedding_model.device)
        batch_size = 32

        # Process embeddings in batches to optimize memory usage
        for i in range(0, len(text_chunks), batch_size):
            batch_embeddings = embedding_model.encode(text_chunks[i:i + batch_size], convert_to_tensor=True)
            embeddings = torch.cat((embeddings, batch_embeddings), dim=0)
            logging.info(f"Processed batch {i // batch_size + 1}/{(len(text_chunks) + batch_size - 1) // batch_size}")

        df = pd.DataFrame(processed_data)
        df["embedding"] = embeddings.cpu().numpy().tolist()
        return df
    except Exception as e:
        logging.error(f"Error creating embeddings: {e}", exc_info=True)
        return pd.DataFrame()


def semantic_search(query: str, embeddings_df: pd.DataFrame, embedding_model: SentenceTransformer, num_results: int) -> List[Dict[str, Any]]:
    """Performs semantic search using embeddings and returns the top results."""
    try:
        # Create embedding for the query
        query_embedding = embedding_model.encode(query, convert_to_tensor=True)

        # Convert embeddings from DataFrame to a tensor
        embeddings = torch.tensor(np.array(embeddings_df["embedding"].tolist()), dtype=torch.float32).to(embedding_model.device)

        # Measure search time
        start_time = timer()
        dot_scores = util.dot_score(query_embedding, embeddings)[0]
        end_time = timer()
        logging.info(f"Time taken to get scores on {len(embeddings)} embeddings: {end_time - start_time:.5f} seconds.")

        # Get the top results
        top_results = torch.topk(dot_scores, k=num_results)
        results = []

        # Format the results
        for score, idx in zip(top_results.values, top_results.indices):
            idx = idx.item()  # Convert tensor to integer
            result = {
                "score": score.item(),
                "text": embeddings_df.iloc[idx]["text"],
                "file_name": embeddings_df.iloc[idx]["file_name"],
                **{k: v for k, v in embeddings_df.iloc[idx].items() if k not in ["text", "file_name", "embedding"]}
            }
            results.append(result)

        return results
    except Exception as e:
        logging.error(f"Error during semantic search: {e}", exc_info=True)
        return []


def create_results_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Creates a DataFrame from search results for better visualization."""
    if not results:
        logging.info("No results to display.")
        return pd.DataFrame()

    # Extract result details
    scores = [result["score"] for result in results]
    texts = [result["text"] for result in results]
    file_names = [result["file_name"] for result in results]

    result_data = {
        "score": scores,
        "text": texts,
        "file_name": file_names
    }

    # Include additional fields from results
    for key in results[0].keys():
        if key not in ["score", "text", "file_name"]:
            result_data[key] = [result[key] for result in results]

    return pd.DataFrame(result_data)
def query_model(prompt: str) -> str:
    """
    Query model via Hugging Face API.
    """
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 1500, 
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True
        }
    }
    
    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # This will raise an exception for 4xx and 5xx status codes
        
        # Print the raw response for debugging
        print("Raw API Response:")
        print(response.text)
        
        response_json = response.json()
        
        # Check if the response is a list (as expected for some models)
        if isinstance(response_json, list) and len(response_json) > 0:
            if 'generated_text' in response_json[0]:
                return response_json[0]['generated_text']
            elif 'summary_text' in response_json[0]:  # BART models often use 'summary_text'
                return response_json[0]['summary_text']
        
        # If it's not a list or doesn't contain the expected keys, return the whole response
        return str(response_json)
    
    except requests.exceptions.RequestException as e:
        error_message = f"API request failed: {str(e)}"
        logging.error(error_message)
        return f"Sorry, I couldn't generate an answer at this time. Error: {error_message}"
    except (KeyError, IndexError, ValueError) as e:
        logging.error(f"Error parsing API response: {str(e)}")
        return f"Sorry, I couldn't parse the model's response. Error: {str(e)}"

def parse_response(response: str) -> str:
    """
    Parse and format the BLOOMZ response.
    """
    # Remove the original prompt from the response
    response = response.split("Answer:", 1)[-1].strip()
    
    # Define regex patterns for different levels of structure
    patterns = [
        (r'•\s*', '\n• '),  # Main points
        (r'\(\s*', '\n  • '),  # Sub-points
        (r'\([i-v]+\)\s*', '\n    • ')  # Sub-sub-points
    ]
    
    # Apply formatting
    formatted_response = response
    for pattern, replacement in patterns:
        formatted_response = re.sub(pattern, replacement, formatted_response)
    
    # Remove any extra newlines
    formatted_response = re.sub(r'\n+', '\n', formatted_response)
    
    return formatted_response.strip()

def main(files: list, query: str) -> None:
    """Main function to process files, create embeddings, perform semantic search, and output results."""
    # Load SentenceTransformer model only once
    embedding_model = SentenceTransformer("all-mpnet-base-v2", device="cuda" if torch.cuda.is_available() else "cpu")

    # Process files
    processed_data = process_files(files)
    if not processed_data:
        logging.error("No data processed. Exiting.")
        return

    # Create embeddings
    embeddings_df = create_embeddings(processed_data, embedding_model)
    if embeddings_df.empty:
        logging.error("No embeddings created. Exiting.")
        return

    # Perform semantic search
    results = semantic_search(query, embeddings_df, embedding_model, num_results=5)
    if not results:
        logging.warning("No results found. Proceeding with empty context.")
        context = ""
    else:
        # Prepare context for BLOOMZ as a single string, limiting to 1000 characters
        context = " ".join([result['text'] for result in results])[:1000]
    
    logging.info(f"Context length: {len(context)} characters")

    prompt = f"""I have a data: '{context}'. Based on this reference, I need an answer to '{query}'. Please combine your internal knowledge with the provided data, but present the answer in a clear, conversational, and engaging way, making it easy and interesting for the user to read."""

    logging.info("Sending request to Hugging Face API")
    response = query_model(prompt)
    logging.info("Received response from Hugging Face API")

    # Parse and format the response
    formatted_response = parse_response(response)

    # Display formatted BLOOMZ response
    print("*"*50+"\nFormatted Answer:")
    print(formatted_response)

    # If the formatted response is empty or doesn't contain the expected structure,
    # display the raw response for debugging
    if not formatted_response or '•' not in formatted_response:
        print("*"*50+"\nRaw Response (for debugging):")
        print(response)

if __name__ == "__main__":
    # Example file input
    files = ['files/text.pdf', 'files/audio.mp3', 'files/image.png']  
    query = "Explain in very detailed way what is protein?"
    main(files, query)