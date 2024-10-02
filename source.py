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
import google.generativeai as genai
import warnings
import json


# Suppress specific FutureWarning messages
warnings.filterwarnings("ignore", category=FutureWarning)


# Load environment variables
load_dotenv()

# Gemini-API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Set up logging
logging.basicConfig(level=logging.INFO)

# Importing processors (assumed to be your custom modules)
from src.text_processor import process_text_file
from src.audio_processor import process_audio_from_url
from src.video_processor import process_video_file
from src.image_processor import process_image_file

from response import generate_response

def process_files(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Processes a list of files in parallel and returns their processed content."""
    
    def process_single_file(file_path):
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()
        file_name = os.path.basename(file_path)
        
        if "?alt=media&token=" in extension:
            extension = list(extension.split("?"))[0]
        print("\nprocessing file type : ",extension)
        try:
            if extension in ['.txt', '.pdf', '.docx']:
                return process_text_file(file_path)
            elif extension in ['.mp3', '.wav', '.flac']:
                return process_audio_from_url(file_path)
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
    try:
        # Process files in parallel, limiting threads to the number of CPU cores
        with ThreadPoolExecutor(max_workers=min(len(file_paths), os.cpu_count())) as executor:
            results = executor.map(process_single_file, file_paths)
        # Flatten the results
        processed_data = [item for result in results for item in result]

        if not processed_data:
            return []
        return processed_data
    except ValueError:
        logging.error("Input list is empty or contains invalid file paths")
        return []

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
    
    
def count_tokens(text: str) -> int:
    """Roughly estimate the number of tokens in a text."""
    return len(text.split())
    

def main(files: list, query: str, min_text_length: int = 500, max_gemini_tokens: int = 7500):
    """Main function to process files, perform semantic search or send data directly to Gemini."""
    
    try:
        noData=False
        # Process files (your existing file processing logic)
        processed_data = process_files(files)
        if not processed_data:
            noData = True
        # Combine all text chunks
        combined_text = " ".join([item["text"] for item in processed_data])

        logging.info(f"Total text length: {len(combined_text)} characters")

        # Count tokens and check if they exceed the allowed limit for Gemini
        token_count = count_tokens(combined_text)

        # If token count is within limits, send directly to Gemini for response generation
        if token_count < min_text_length:
            logging.info(f"Text is below the threshold ({min_text_length} tokens). Sending directly to Gemini.")
            response = generate_response(combined_text, query,noData)
            return response
        else:
            logging.info(f"Text exceeds the maximum allowed tokens ({max_gemini_tokens}). Performing semantic search.")
            # Only initialize embeddings when needed
            embedding_model = SentenceTransformer("all-mpnet-base-v2", device="cuda" if torch.cuda.is_available() else "cpu")

            # Create embeddings
            embeddings_df = create_embeddings(processed_data, embedding_model)
            if embeddings_df.empty:
                logging.error("No embeddings created. Exiting.")
                return {"error": "Failed to create embeddings from the processed data."}

            # Perform semantic search
            num_results = min(1, len(embeddings_df))  # Adjust number of results based on available data
            results = semantic_search(query, embeddings_df, embedding_model, num_results)
            print("Semantic Searchs return the top results with relevant scores and contextual information. \n",results)
            if not results:
                logging.error("No results found. Exiting.")
                return {"error": "Semantic search returned no results."}
            context = " ".join([result['text'] for result in results])  # Example context generation from results
            response = generate_response(context, query, noData)
            return response
    except Exception as e:
        logging.error(f"Error: {e}")
        return {"error": "An error occurred during the main process."}

if __name__ == "__main__":
    files = [
        # Your file paths go here
    ] 
    query = "Introduce yourself, what are you?"
    main(files, query)