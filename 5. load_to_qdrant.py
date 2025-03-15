import os
import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import json
import ast
from tqdm import tqdm

# Configuration
FOLDER_PATH = "D:\MasterAI\Math4AI\do_an_2\Embed\KB_raw"  # Update this path
COLLECTION_NAME = "vietnamese_knowledge_base"
VECTOR_SIZE = None  # Will be determined from the first file
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

def parse_embedding(embedding_str):
    """Parse embedding string to a list of floats."""
    try:
        # Try parsing as a JSON string first
        return json.loads(embedding_str)
    except json.JSONDecodeError:
        try:
            # Try parsing as a Python literal (list or array string)
            return ast.literal_eval(embedding_str)
        except (SyntaxError, ValueError):
            # Handle case where embedding might be stored differently
            raise ValueError(f"Could not parse embedding: {embedding_str[:50]}...")

def initialize_qdrant_collection(client, vector_size):
    """Initialize Qdrant collection if it doesn't exist."""
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if COLLECTION_NAME not in collection_names:
        print(f"Creating new collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"Collection {COLLECTION_NAME} created successfully")
    else:
        print(f"Collection {COLLECTION_NAME} already exists")

def process_csv_files(folder_path):
    """Process all CSV files in the given folder."""
    # Connect to Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Get list of all CSV files
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    id = 0
    # Process each CSV file
    global VECTOR_SIZE
    for file_index, csv_file in enumerate(csv_files):
        file_path = os.path.join(folder_path, csv_file)
        print(f"Processing file {file_index + 1}/{len(csv_files)}: {csv_file}")
        
        try:
            # Read CSV into DataFrame
            df = pd.read_csv(file_path)
            
            # Verify required columns exist
            required_columns = ['subject_id', 'subject', 'con', 're', 'embedding']
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                print(f"Warning: File {csv_file} is missing required columns: {missing_columns}")
                continue
            
            # Sample one embedding to determine vector size if not already set
            if VECTOR_SIZE is None:
                sample_embedding = parse_embedding(df.iloc[0]['embedding'])
                VECTOR_SIZE = len(sample_embedding)
                print(f"Determined vector size: {VECTOR_SIZE}")
                
                # Initialize collection with proper vector size
                initialize_qdrant_collection(client, VECTOR_SIZE)
            
            # Prepare points for batch upload
            points = []
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
                try:
                    embedding_vector = parse_embedding(row['embedding'])
                    
                    # Ensure embedding has correct dimensionality
                    if len(embedding_vector) != VECTOR_SIZE:
                        print(f"Warning: Embedding dimension mismatch. Expected {VECTOR_SIZE}, got {len(embedding_vector)}")
                        continue
                    
                    # Create a point for Qdrant
                    point = PointStruct(
                        id=id,
                        vector=embedding_vector,
                        payload={
                            'subject_id': row['subject_id'],
                            'subject': row['subject'],
                            'con': row['con'],
                            're': row['re']
                        }
                    )
                    points.append(point)
                    id += 1
                    
                except Exception as e:
                    print(f"Error processing row: {e}")
            
            # Upload points in batches
            BATCH_SIZE = 100
            for i in range(0, len(points), BATCH_SIZE):
                batch = points[i:i + BATCH_SIZE]
                try:
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=batch
                    )
                except Exception as e:
                    print(f"Error uploading batch: {e}")
            
            print(f"Successfully processed {len(points)} entries from {csv_file}")
            
        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")
    
    print("All files processed. Data loaded into Qdrant.")

if __name__ == "__main__":
    process_csv_files(FOLDER_PATH)