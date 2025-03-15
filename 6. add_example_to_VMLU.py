import json
import os
from qdrant_client import QdrantClient
from tqdm import tqdm
import pandas as pd

# Configuration
JSONL_FILE_PATH = "D:/MasterAI/Math4AI/do_an_2/vlmu_mqa_v1.5/test.jsonl"
OUTPUT_FILE_PATH = "D:/MasterAI/Math4AI/do_an_2/vlmu_mqa_v1.5/test_added_example.jsonl"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "vietnamese_knowledge_base"
OPENAI_API_KEY = "xxxx"

from openai import OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Connect to Qdrant
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def get_embedding(text):

    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    return response.data[0].embedding

def find_relevant_examples(query_embedding, top_k=3):
    """Find relevant examples in Qdrant using the query embedding."""
    try:
        search_results = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_embedding,
            limit=top_k
        )
        
        examples = []
        for i, result in enumerate(search_results):
            con = result.payload.get('con', '')
            re = result.payload.get('re', '')
            examples.append(f"{i+1}. Nếu {con} thì {re}")
        
        return "\n".join(examples)
    except Exception as e:
        print(f"Error searching Qdrant: {e}")
        return ""

def process_jsonl_file(index_start):
    """Process the JSONL file starting from index_start and handle up to 1000 items."""
    # Read the JSONL file into a pandas DataFrame
    df = pd.read_json(JSONL_FILE_PATH, lines=True)
    
    # Calculate end index (ensure we don't go beyond the DataFrame's length)
    total_items = len(df)
    index_end = min(index_start + 1000, total_items)
    
    print(f"Processing items from index {index_start} to {index_end-1} (total: {index_end-index_start})")
    
    # Get the subset to process
    subset_df = df.iloc[index_start:index_end].copy()
    
    # Process each item in the subset
    for idx, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc="Processing items"):
        question = row['question']
        
        # Get embedding for the question
        embedding = get_embedding(question)
        if embedding is None:
            print(f"Skipping item {row['id']} due to embedding error")
            subset_df.at[idx, 'example'] = ""
            continue
        
        # Find relevant examples
        examples = find_relevant_examples(embedding)
        
        # Add examples to the DataFrame
        subset_df.at[idx, 'example'] = examples

    out_PATH = "D:/MasterAI/Math4AI/do_an_2/vlmu_mqa_v1.5/test_added_example.jsonl" + '_' + str(index_start) + ".jsonl"
    # If output file exists and we're not starting from the beginning, append to it
    if os.path.exists(out_PATH):
        mode = 'a'  # Append mode
    else:
        mode = 'w'  # Write mode (create new file)
    
    # Write the processed subset to the output file
    with open(out_PATH, mode, encoding='utf-8') as file:
        for _, row in subset_df.iterrows():
            file.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\n')
    
    print(f"Successfully processed {len(subset_df)} items and saved to {out_PATH}")
    print(f"Next batch should start at index {index_end}")
    
    return index_end  # Return the next index to start from
    
    # Write the updated items to a new JSONL file
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Successfully processed {len(items)} items and saved to {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--Output", help = "Show Output")

    # Read arguments from command line
    args = parser.parse_args()

    process_jsonl_file(int(args.Output))