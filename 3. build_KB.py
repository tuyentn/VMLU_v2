#%%

import os
import json
import time
import openai
import pandas as pd
from tqdm import tqdm
import argparse
import re

# Set up OpenAI API key
openai.api_key = "xxx"
if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

from openai import OpenAI
client = OpenAI(api_key=openai.api_key)


#%%
# Create subjects DataFrame
def create_subjects_dataframe():
    data = [
        # STEM subjects
        [1, "Elementary Mathematics", "STEM"],
        [2, "Elementary Science", "STEM"],
        [3, "Middle School Biology", "STEM"],
        [4, "Middle School Chemistry", "STEM"],
        [5, "Middle School Mathematics", "STEM"],
        [6, "Middle School Physics", "STEM"],
        [7, "High School Biology", "STEM"],
        [8, "High School Chemistry", "STEM"],
        [9, "High School Mathematics", "STEM"],
        [10, "High School Physics", "STEM"],
        [11, "Applied Informatics", "STEM"],
        [12, "Computer Architecture", "STEM"],
        [13, "Computer Network", "STEM"],
        [14, "Discrete Mathematics", "STEM"],
        [15, "Electrical Engineering", "STEM"],
        [16, "Introduction to Chemistry", "STEM"],
        [17, "Introduction to Physics", "STEM"],
        [18, "Introduction to Programming", "STEM"],
        [19, "Metrology Engineer", "STEM"],
        [20, "Operating System", "STEM"],
        [21, "Statistics and Probability", "STEM"],
        
        # Social Science subjects
        [22, "Middle School Civil Education", "Social Science"],
        [23, "Middle School Geography", "Social Science"],
        [24, "High School Civil Education", "Social Science"],
        [25, "High School Geography", "Social Science"],
        [26, "Business Administration", "Social Science"],
        [27, "Ho Chi Minh Ideology", "Social Science"],
        [28, "Macroeconomics", "Social Science"],
        [29, "Microeconomics", "Social Science"],
        [30, "Principles of Marxism and Leninism", "Social Science"],
        [31, "Sociology", "Social Science"],
        
        # Humanity subjects
        [32, "Elementary History", "Humanity"],
        [33, "Middle School History", "Humanity"],
        [34, "Middle School Literature", "Humanity"],
        [35, "High School History", "Humanity"],
        [36, "High School Literature", "Humanity"],
        [37, "Administrative Law", "Humanity"],
        [38, "Business Law", "Humanity"],
        [39, "Civil Law", "Humanity"],
        [40, "Criminal Law", "Humanity"],
        [41, "Economic Law", "Humanity"],
        [42, "Education Law", "Humanity"],
        [43, "History of World Civilization", "Humanity"],
        [44, "Idealogical and Moral Cultivation", "Humanity"],
        [45, "Introduction to Laws", "Humanity"],
        [46, "Introduction to Vietnam Culture", "Humanity"],
        [47, "Logic", "Humanity"],
        [48, "Revolutionary Policy of the Vietnamese Commununist Part", "Humanity"],
        [49, "Vietnamese Language and Literature", "Humanity"],
        
        # Other subjects
        [50, "Accountant", "Others"],
        [51, "Clinical Pharmacology", "Others"],
        [52, "Environmental Engineering", "Others"],
        [53, "Internal Basic Medicine", "Others"],
        [54, "Preschool Pedagogy", "Others"],
        [55, "Tax Accountant", "Others"],
        [56, "Tax Civil Servant", "Others"],
        [57, "Civil Servant", "Others"],
        [58, "Driving License Certificate", "Others"]
    ]
    
    return pd.DataFrame(data, columns=["id_subject", "subject", "domain"])

# Get domains from DataFrame
def get_domains(df):
    return df['domain'].unique().tolist()

# Get subjects for a domain
def get_subjects_for_domain(df, domain):
    return df[df['domain'] == domain]['subject'].tolist()

# Get domain for a subject
def get_domain_for_subject(df, subject):
    matches = df[df['subject'] == subject]
    if len(matches) > 0:
        return matches.iloc[0]['domain']
    return None

# Get subject ID
def get_subject_id(df, subject):
    matches = df[df['subject'] == subject]
    if len(matches) > 0:
        return matches.iloc[0]['id_subject']
    return None

# Example rules for different subjects (in Vietnamese)
EXAMPLES = {
    "Elementary Mathematics": [
        {"con": "số chia hết cho 2", "re": "số đó là số chẵn"},
        {"con": "a + b = b + a", "re": "phép cộng có tính chất giao hoán"},
        {"con": "hai số có tổng bằng 0", "re": "hai số đó đối nhau"}
    ],
    "High School Physics": [
        {"con": "lực tác dụng lên vật tăng", "re": "gia tốc của vật tăng"},
        {"con": "nhiệt độ của chất khí tăng và thể tích không đổi", "re": "áp suất tăng"},
        {"con": "vật chuyển động với vận tốc không đổi", "re": "gia tốc bằng 0"}
    ],
    "High School Biology": [
        {"con": "thực vật không nhận được ánh sáng", "re": "không thể quang hợp"},
        {"con": "tế bào không có ty thể", "re": "không thể thực hiện hô hấp tế bào hiếu khí"},
        {"con": "gene bị đột biến", "re": "protein được tổng hợp có thể bị thay đổi"}
    ]
    # Add more examples for other subjects as needed
}

def get_default_examples(subject):
    """Return default examples for a subject or general examples if none exist"""
    if subject in EXAMPLES:
        return EXAMPLES[subject]
    return [
        {"con": "A", "re": "B"},
        {"con": "điều kiện X xảy ra", "re": "kết quả Y sẽ xuất hiện"},
        {"con": "nguyên nhân P tồn tại", "re": "hệ quả Q sẽ theo sau"}
    ]

def format_examples_for_prompt(examples):
    """Format example objects for the prompt"""
    formatted = []
    for ex in examples:
        formatted.append(f"Nếu {ex['con']} thì {ex['re']}")
    return formatted

def parse_rule_string(rule_string):
    """Parse a rule string in the format 'Nếu [condition] thì [result]' into a dict"""
    # Remove leading numbers or dashes if they exist
    rule_string = re.sub(r'^[\d\-\.\s]+', '', rule_string.strip())
    
    # Extract condition and result using regex
    match = re.match(r'(?:Nếu\s+)?(.+?)\s+thì\s+(.+)', rule_string, re.IGNORECASE)
    
    if match:
        condition, result = match.groups()
        return {"con": condition.strip(), "re": result.strip()}
    else:
        # If regex fails, try to split on "thì"
        parts = rule_string.split("thì", 1)
        if len(parts) == 2:
            condition = parts[0].replace("Nếu", "", 1).strip()
            result = parts[1].strip()
            return {"con": condition, "re": result}
    
    # Return the whole string as condition if parsing fails
    return {"con": rule_string, "re": ""}

def generate_rules(subject, num_rules=500, model="gpt-4-turbo", examples=None):
    """Generate knowledge rules for a given subject using OpenAI API"""
    if examples is None:
        examples = get_default_examples(subject)
    
    # Format examples for the prompt
    examples_text = "\n".join(format_examples_for_prompt(examples))
    
    # Format structured JSON examples
    json_examples = json.dumps(examples[:2], ensure_ascii=False, indent=2)
    
    prompt = f"""Hãy tạo ra {num_rules} quy tắc kiến thức đúng đắn trong lĩnh vực {subject} bằng tiếng Việt.
Mỗi quy tắc phải có dạng "Nếu [điều kiện] thì [kết quả]".
Các quy tắc phải chính xác về mặt học thuật, đa dạng và bao quát nhiều khía cạnh của {subject}.

Dưới đây là một số ví dụ về định dạng quy tắc:
{examples_text}

Nhưng tôi muốn kết quả trả về là một mảng JSON với mỗi phần tử có cấu trúc:
{{
  "con": "điều kiện (không bao gồm từ 'Nếu')",
  "re": "kết quả đúng (không bao gồm từ 'thì')"
}}

Ví dụ:
{json_examples}

Trả về kết quả dưới dạng mảng JSON, mỗi phần tử là một đối tượng có hai trường "con" và "re". Kết quả trả về cũng bao gồm ví dụ trên.
"""

    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Bạn là một chuyên gia về lĩnh vực kiến thức và giáo dục, đặc biệt giỏi trong việc tạo ra các quy tắc kiến thức chính xác."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=5000
            )

            content = completion.choices[0].message.content
            
            # Extract JSON array from response
            try:
                # Try to find JSON array in the response
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    rules = json.loads(json_str)
                    
                    # Convert any old format to new format if needed
                    for i, rule in enumerate(rules):
                        if "condition" in rule and "con" not in rule:
                            rules[i] = {"con": rule["condition"], "re": rule["result"]}
                    
                    return rules
                else:
                    # If no JSON array found, try to parse the whole response
                    rules = json.loads(content)
                    
                    # Convert any old format to new format if needed
                    for i, rule in enumerate(rules):
                        if "condition" in rule and "con" not in rule:
                            rules[i] = {"con": rule["condition"], "re": rule["result"]}
                    
                    return rules
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract rules from text
                rules = []
                lines = content.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("Nếu") and "thì" in line:
                        rule_dict = parse_rule_string(line)
                        rules.append(rule_dict)
                
                if rules:
                    return rules
                else:
                    print(f"Could not parse response for {subject}. Retrying...")
                    print(f"Response snippet: {content[:200]}...")
                    time.sleep(retry_delay)
                    continue
                
        except Exception as e:
            print(f"Error generating rules for {subject}: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to generate rules for {subject} after {max_retries} attempts.")
                return []
    
    return []

def save_rules(domain, subject, subject_id, rules, output_dir="knowledge_base"):
    """Save generated rules to a JSON file"""
    # Create output directory if it doesn't exist
    domain_dir = os.path.join(output_dir, domain)
    os.makedirs(domain_dir, exist_ok=True)
    
    # Create filename with ID prefix
    filename = os.path.join(domain_dir, f"{subject_id}_{subject.replace(' ', '_')}.json")
    
    # Save rules to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(rules)} rules for {subject} (ID: {subject_id}) to {filename}")

def load_examples_from_jsonl(jsonl_file):
    """Load questions from a JSONL file as examples for rule generation"""
    examples_by_subject_id = {}
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # Extract subject ID from the first two numbers in the question ID
                subject_id = int(item['id'].split('-')[0])
                
                # Create a rule from the question and answer
                question = item['question']
                answer_letter = item['answer'] if 'answer' in item else ' '
                answer_text = next((c for c in item['choices'] if c.startswith(answer_letter)), "")
                answer_content = answer_text.split(". ", 1)[1] if ". " in answer_text else answer_text[2:]
                if answer_content == "":
                    answer_content = "Câu trả lời đúng"
                # Format as a condition-result pair
                rule = {
                    "con": question,
                    "re": answer_content
                }
                
                # Add to examples dictionary
                if subject_id not in examples_by_subject_id:
                    examples_by_subject_id[subject_id] = []
                examples_by_subject_id[subject_id].append(rule)
    
    except Exception as e:
        print(f"Error loading examples from {jsonl_file}: {str(e)}")
        return {}
    
    return examples_by_subject_id

def get_examples_for_subject(subject_id, examples_by_subject_id, default_examples):
    """Get examples for a subject from loaded examples or use defaults"""
    if subject_id in examples_by_subject_id and examples_by_subject_id[subject_id]:
        return examples_by_subject_id[subject_id]
    return default_examples

#%%
subjects_df = create_subjects_dataframe()
all_examples = load_examples_from_jsonl('vlmu_mqa_v1.5/test.jsonl')

#%%

print(all_examples[1][0])


#%%
#print each rule in examples_by_subject_id
total_subjects = len(subjects_df)
for domain in get_domains(subjects_df):
    domain_subjects = get_subjects_for_domain(subjects_df, domain)
    print(f"\nProcessing {domain} ({len(domain_subjects)} subjects)...")
    for subject in tqdm(domain_subjects, desc=f"Processing {domain}"):
        print('Subject: ', subject)
        rules = []
        subject_id = get_subject_id(subjects_df, subject)
        examples = get_examples_for_subject(subject_id, all_examples, "no")
        if examples == 'no':
            print("No examples found")
        else:
            for index_e, example in enumerate(examples):
                print(str(index_e) + '. ' + str(example['con'])[:20])
                rules5 = generate_rules(subject, num_rules=10, model="gpt-4o-mini", examples=[example])
                rules = rules + rules5
            save_rules(domain, subject, subject_id, rules, output_dir='knowledge_base')



# %%
