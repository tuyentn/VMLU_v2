import os
import json
import pandas as pd
import re
from tqdm import tqdm

def embed_KB(raw_text):
    from openai import OpenAI
    client = OpenAI(api_key="xxx")

    response = client.embeddings.create(
        input=raw_text,
        model="text-embedding-3-small"
    )

    return response.data[0].embedding


def process_jsonl_files_recursively(master_folder):
    # List to store all data
    all_data = []
    
    # Walk through all subdirectories recursively
    for root, dirs, files in os.walk(master_folder):
        # Process each JSON file in the current directory
        for filename in files:
            if filename.endswith('.json') or filename.endswith('.jsonl'):
                file_path = os.path.join(root, filename)
                
                # Extract subject_id and subject name from filename
                match = re.match(r"(\d+)_(.+)\.json", filename)
                if match:
                    subject_id = match.group(1)
                    subject = match.group(2).replace('_', ' ')  # Replace underscores with spaces
                else:
                    # Fallback if filename doesn't match expected pattern
                    subject_id = ""
                    subject = filename.split('.')[0]  # Remove extension
                
                # Read the JSON file
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        try:
                            # Each file contains a JSON array with multiple entries
                            data = json.load(file)
                            
                            # Add each entry with the subject_id, subject, con, and re
                            for entry in data:
                                all_data.append({
                                    'subject_id': subject_id,
                                    'subject': subject,
                                    'con': entry.get('con', ''),
                                    're': entry.get('re', '')
                                })
                        except json.JSONDecodeError as e:
                            print(f"Error decoding {file_path}: {e}")
                except Exception as e:
                    print(f"Error opening {file_path}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    return df

def main(start_index = 0):
    # # Specify the master folder path
    # master_folder = "C:/Users/tuyen.tranngoc/learning/MasterAI/knowledge_base"  # Replace with your actual master folder path
    
    # # Process all files recursively
    # df = process_jsonl_files_recursively(master_folder)

    
    # # remove rows with re column = 'Câu trả lời đúng'
    # df = df[df['re'] != 'Câu trả lời đúng']

    print("Start embedding from index: ", start_index)

    df = pd.read_csv("subjects_data.csv")
    # from start_index to start_index + 10000
    df = df[start_index:start_index+10000]

    df['embedding'] = None
    
    # Process in batches
    for idx, row in tqdm(df.iterrows(), desc="Generating embeddings", total=len(df)):
        df.at[idx, 'embedding'] = embed_KB(row['con'])
        if (idx+1) % 1000 == 0:
            df.to_csv("embed_KB/subjects_data_"+str(start_index)+"_"+str(idx//1000)+".csv", index=False, encoding='utf-8')
        


    # Export to CSV
    # output_file = "subjects_data.csv"
    # df.to_csv(output_file, index=False, encoding='utf-8')

    # print(f"Processed {len(df)} entries from {df['subject'].nunique()} subjects")
    # print(f"Found data in {df['subject_id'].nunique()} unique subject IDs")
    # print(f"Data saved to {output_file}")

def test(a):
    print(a)
    return None

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--Output", help = "Show Output")

    # Read arguments from command line
    args = parser.parse_args()


    main(int(args.Output))
    # test(args.Output)