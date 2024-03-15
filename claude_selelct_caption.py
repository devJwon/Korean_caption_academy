import os
import pandas as pd
import random
import json
from dotenv import load_dotenv
import logging
import httpx
import base64
import anthropic

load_dotenv()
api_key = os.getenv('ANTHROPIC_API_KEY')
if not api_key:
    raise ValueError("API key not found. Please set the ANTHROPIC_API_KEY in the .env file.")

def fetch_and_encode_image(url):
    response = httpx.get(url, follow_redirects=True)
    if response.status_code == 200:
        return base64.b64encode(response.content).decode('utf-8')
    else:
        logging.error(f"Failed to fetch image from {url}")
        return None

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

def get_image_caption(client, image_data, captions, language='en'):
    prompt = f"""
[Task] Choose the caption that best describes the given picture

[Output Format]
Return the chosen caption in JSON format: {{selected caption number:'caption text'}}.

[captions]
1: {captions[0]}
2: {captions[1]}
3: {captions[2]}

[Best caption]
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
        # Assuming the client can handle base64 encoded images directly; adjust as necessary.
        response = client.messages.create(
            model="claude-3-opus-20240229",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        logging.info(f"Sending request to Anthropic with base64 encoded image and captions: {captions}")
        print(response.content)
        response_text = response.content[0].text
        logging.info("Received response from Anthropic")
        logging.info(f"Response: {response_text}")
        return response_text
    except Exception as e:
        logging.error(f"Error processing image. Error: {e}")
        return json.dumps({"error": str(e)})

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    lang = 'en'
    # lang = 'ko'

    # Use your anthropic client initialization here
    client = anthropic.Anthropic(api_key=api_key)  # Adjusted to use the Anthropics client with an API key
    error_rows = []
    logging.info("Loading image data from CSV files...")
    csv_file_paths = ['data/desc/ed_0.csv', 'data/desc/ed_1.csv', 'data/desc/ed_2.csv', 'data/desc/ed_3.csv']
    combined_data = load_image_data_from_csvs(csv_file_paths)
    combined_data[['caption1', 'caption2', 'caption3']] = combined_data.apply(lambda row: pd.Series(shuffle_captions(row)), axis=1)
    
    results_dir = 'result/desc'
    os.makedirs(results_dir, exist_ok=True)
    results_file_path = os.path.join(results_dir, f'claude_{lang}_desc.csv')
    
    logging.info(f"Processing images for humor evaluation...")
    for index, row in combined_data.iterrows():
        logging.info(f"Processing image {index + 1}/{len(combined_data)}...")
        image_data = fetch_and_encode_image(row['url'])
        if image_data:
            captions = [row['caption1'], row['caption2'], row['caption3']]
            response_text = get_image_caption(client, image_data, captions, lang)
            if "error" not in response_text:
                combined_data.at[index, 'response_text'] = response_text
            else:
                error_rows.append(index)
                logging.error(f"Error processing row {index}: {response_text}")
        else:
            error_rows.append(index)
            logging.error(f"Error fetching or encoding image for row {index}")

    combined_data.to_csv(results_file_path, index=False, encoding='utf-8')
    logging.info(f"Responses have been saved to {results_file_path}")
    if error_rows:
        logging.error(f"Errors occurred in the following rows: {error_rows}")

if __name__ == "__main__":
    main()
