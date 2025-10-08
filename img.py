import os
import csv
import requests
from concurrent.futures import ThreadPoolExecutor

CSV_FILE = "img.csv"
MAX_WORKERS = 5  # Number of parallel downloads

def sanitize(text):
    if not text:
        return "No_Name"
    return text.replace('/', '_').replace('\\', '_').replace('+', '_').strip()

def download_image(row, idx):
    query = sanitize(row.get('query'))
    name = sanitize(row.get('name'))
    img_url = row.get('photo_p')  # you can change to 'photo_url_big' or 'original_photo_url'

    if not img_url:
        print(f"Skipping empty URL for {query}/{name}")
        return

    folder_path = os.path.join(query, name)
    os.makedirs(folder_path, exist_ok=True)

    img_path = os.path.join(folder_path, f"img_{idx+1}.jpg")

    try:
        response = requests.get(img_url, timeout=15)
        response.raise_for_status()
        with open(img_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {img_path}")
    except Exception as e:
        print(f"Failed to download {img_url}: {e}")

def main():
    with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        rows = list(reader)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for idx, row in enumerate(rows):
            executor.submit(download_image, row, idx)

if __name__ == "__main__":
    main()
