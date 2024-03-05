from openai import OpenAI
import os
from dotenv import load_dotenv
import csv
import random
import chardet  # Make sure to install chardet with pip install chardet

load_dotenv()

# Access your API key
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY in the .env file.")

# Load the image URL and captions from a CSV file, and shuffle the captions
def load_image_data_from_csv(csv_file_path):
    data = []
    encodings_to_try = ['utf-8', 'ISO-8859-1', 'cp1252']  # Common encodings
    
    for encoding in encodings_to_try:
        try:
            with open(csv_file_path, mode='r', newline='', encoding=encoding) as file:
                reader = csv.DictReader(file)
                for row in reader:
                    captions = [row['caption1'], row['caption2'], row['caption3']]
                    random.shuffle(captions)  # Shuffle the captions randomly
                    data.append({'url': row['url'], 'captions': captions})
                return data  # Successfully read data, break out of the loop
        except UnicodeDecodeError:
            print(f"Failed to decode with {encoding}, trying next encoding.")
    
    # If all specified encodings fail, attempt to detect encoding
    print("Attempting to detect file encoding...")
    detected_encoding = chardet.detect(open(csv_file_path, 'rb').read())['encoding']
    print(f"Detected encoding: {detected_encoding}")
    
    try:
        with open(csv_file_path, mode='r', newline='', encoding=detected_encoding) as file:
            reader = csv.DictReader(file)
            for row in reader:
                captions = [row['caption1'], row['caption2'], row['caption3']]
                random.shuffle(captions)
                data.append({'url': row['url'], 'captions': captions})
    except Exception as e:
        print(f"Failed to read file with detected encoding {detected_encoding}: {e}")
        raise  # Re-raise the exception if reading still fails
    
    return data

def get_image_caption(image_url, captions):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""
[과제] 제공된 이미지와 관련된 가장 재미있는 제목을 선택합니다. 선택한 제목은 이미지를 유머러스하게 설명해야 하며 이미지의 맥락과 관련이 있어야 합니다.

[지침]
- 제공된 각 제목의 유머를 평가합니다.
- 제목이 이미지와 직접 관련이 있는지 확인합니다.
- 유머 없이 단순히 이미지를 설명하는 제목은 피하세요.
- 가장 유머러스하고 관련성이 높은 제목을 선택하세요.

[출력 형식]
선택한 제목을 JSON 형식으로 반환합니다: {{'선택된 제목':'여기에 제목 텍스트'}}.

[제목]
1: {captions[0]}
2: {captions[1]}
3: {captions[2]}

[가장 재미있는 제목 선택]
                     """},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content

csv_file_path = 'data/korean_caption_total_caption2_shuffle.csv'
image_urls = load_image_data_from_csv(csv_file_path)

# Prepare the results directory and file
results_dir = 'result'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

results_file_path = os.path.join(results_dir, 'humor_ko_result.csv')

# Process each URL and write results to a CSV file
with open(results_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["URL", "Answer"])  # Write the header row

    for item in image_urls:  # Assuming image_urls contains dicts with 'url' and 'captions'
        answer = get_image_caption(item['url'], item['captions'])
        writer.writerow([item['url'], answer])  # Adjusted to match the expected input
        print(f"Processed URL: {item['url']} - Answer: {answer}")


print(f"Answers have been saved to {results_file_path}")