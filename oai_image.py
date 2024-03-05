from openai import OpenAI
import os
from dotenv import load_dotenv
import csv

load_dotenv()

# Access your API key
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY in the .env file.")

# Load the image URL from a CSV file
def load_image_urls_from_csv(csv_file_path):
    urls = []
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            urls.append(row['url'])
    return urls

def get_image_caption(image_url):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "그림을 보고 한 문장으로 웃긴 제목을 지어줘"},
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

csv_file_path = 'images/korean_image.csv'
image_urls = load_image_urls_from_csv(csv_file_path)

# Prepare the results directory and file
results_dir = 'result'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

results_file_path = os.path.join(results_dir, 'humor_caption_ko.csv')

# Process each URL and write results to a CSV file
with open(results_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["URL", "Caption"])  # Write the header row

    for image_url in image_urls:
        caption = get_image_caption(image_url)
        writer.writerow([image_url, caption])  # Write the URL and its caption
        print(f"Processed URL: {image_url} - Caption: {caption}")

print(f"Captions have been saved to {results_file_path}")