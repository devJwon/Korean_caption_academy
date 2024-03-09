import httpx

url = "https://github.com/devJwon/humor-image-captioning/blob/main/images/ny_image/image_127.jpeg?raw=true"

# Making a GET request
response = httpx.get(url, follow_redirects=True)

if response.status_code == 200:
    print("Successfully fetched the content.")
    # You can now access response.content or response.text
else:
    print(f"Failed to fetch the content. Status code: {response.status_code}")
