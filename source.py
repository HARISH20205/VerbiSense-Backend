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
import re

# Load environment variables
load_dotenv()

#Gemini-API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Set up logging
logging.basicConfig(level=logging.INFO)

# Importing processors (assumed to be your custom modules)
from src.text_processor import process_text_file
from src.audio_processor import process_audio_from_url
from src.video_processor import process_video_file
from src.image_processor import process_image_file


def process_files(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Processes a list of files in parallel and returns their processed content."""
    
    def process_single_file(file_path):
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()
        file_name = os.path.basename(file_path)

        # Validate if file exists
        # if not os.path.exists(file_path) or not os.path.isfile(file_path):
        #     logging.error(f"File {file_name} does not exist.")
        #     return []

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
    results = semantic_search(query, embeddings_df, embedding_model, num_results=3)
    if not results:
        logging.error("No results found. Exiting.")
        return

    # Save results to file (as CSV for better structure)
    try:
        df_results = create_results_df(results)
        logging.info("Results saved to results.csv")
    except Exception as e:
        logging.error(f"Error writing results to file: {e}", exc_info=True)

    # Display results in DataFrame
    logging.info("Displaying search results:")
    print(df_results)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    context = " ".join([result['text'] for result in results])
    print("*"*50+"\n"+context)
    
    logging.info(f"Context length: {len(context)} characters")

    prompt = f"""Based on the following context: '{context}', please provide a comprehensive answer to the question: '{query}'.
    Include the following sections in your response:
    1. A brief, direct answer to the question (2-3 sentences)
    2. Key points or facts related to the question (3-5 points)
    3. A short conclusion or summary (1-2 sentences)
    Present the information in a clear, engaging, and easy-to-read manner."""
    
    response = model.generate_content(prompt)
    
    # Print the formatted output
    print("\n" + "="*50)
    print(response.text)
    print("="*50)

if __name__ == "__main__":
    # Example file input
    files = ["https://storage.googleapis.com/verbisense.appspot.com/uploads/vsample.mp4"]  
    query = "What is video about?"
    main(files, query)