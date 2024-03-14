import os
import pandas as pd
import random
import json
from openai import OpenAI
from dotenv import load_dotenv
import logging

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY in the .env file.")


def load_image_data_from_csvs(csv_file_paths):
    data_frames = []
    for csv_file_path in csv_file_paths:
        df = pd.read_csv(csv_file_path)
        data_frames.append(df)
    combined_data = pd.concat(data_frames, ignore_index=False, axis=1)
    return combined_data

def shuffle_captions(row):
    captions = [row['real_description'], row['random_description'], row['humor_caption']]
    random.shuffle(captions)
    return captions


def get_image_caption(client, image_url, captions, language='en'):
    prompt = f"""
[Task] Choose the caption that best describes the given picture

[Output Format]
Return the chosen caption in JSON format: {{selected caption number:'caption text'}}.

[captions]
1: {captions[0]}
2: {captions[1]}
3: {captions[2]}

[Choose the besr caption]
    """ if language == 'en' else f"""
[과제] 주어진 그림을 가장 잘 설명하고 있는 caption을 고르세요.

[출력 형식]
선택한 caption을 JSON 형식으로 반환하세요 : {{'caption 번호':'caption 텍스트'}}.

[caption]
1: {captions[0]}
2: {captions[1]}
3: {captions[2]}

[가장 잘 설명한 caption]
    """      
    try:
        client = OpenAI() 
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        logging.info(f"Sending request to GPT-4 with image URL: {image_url} and captions: {captions}")

        response_text = response.choices[0].message.content.strip()

        logging.info("Received response from GPT-4")
        logging.info(f"Response: {response_text}")

        return response_text
    
    except Exception as e:
        logging.error(f"Error for image URL: {image_url}. Error: {e}")
        return json.dumps({"error": str(e)}) 

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    client = OpenAI(api_key=api_key)
    error_rows = []  # List to track rows where an error occurred
    logging.info("Loading image data from CSV files...")
    csv_file_paths = ['data/desc/kd_0.csv', 'data/desc/kd_1.csv', 'data/desc/kd_2.csv', 'data/desc/kd_3.csv']
    combined_data = load_image_data_from_csvs(csv_file_paths)
    combined_data[['caption1', 'caption2', 'caption3']] = combined_data.apply(lambda row: pd.Series(shuffle_captions(row)), axis=1)
    
    results_dir = 'result'
    os.makedirs(results_dir, exist_ok=True)
    results_file_path = os.path.join(results_dir, 'gpt_ko_desc.csv')
    
    logging.info(f"Processing images for humor evaluation...")
    for index, row in combined_data.iterrows():
        logging.info(f"Processing image {index + 1}/{len(combined_data)}...")
        captions = [row['caption1'], row['caption2'], row['caption3']]
        response_text = get_image_caption(client, row['url'], captions, 'ko')

        if "error" in response_text:
            error_rows.append(index)  # Store the index of the row where an error occurred
            logging.error(f"Error processing row {index}: {response_text}")
            continue  # Skip to the next row

        # Correctly assign the response text to the current row's 'response_text' field
        combined_data.at[index, 'response_text'] = response_text

    combined_data.to_csv(results_file_path, index=False, encoding='utf-8')
    
    logging.info(f"Responses have been saved to {results_file_path}")
    if error_rows:
        logging.error(f"Errors occurred in the following rows: {error_rows}")

if __name__ == "__main__":
    main()
