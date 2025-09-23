import os
import requests
from bs4 import BeautifulSoup

# URL of the JFK 2025 release page
JFK_URL = "https://www.archives.gov/research/jfk/release-2025"
DOWNLOAD_FOLDER = "jfk_2025_pdfs"

# Create folder if it doesn't exist
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

def get_pdf_links(url):
    """Fetch all PDF links from the given URL."""
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve the webpage.")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    pdf_links = []
    
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.endswith(".pdf"):
            full_url = href if href.startswith("http") else f"https://www.archives.gov{href}"
            pdf_links.append(full_url)
    
    return pdf_links

def download_pdfs(pdf_links):
    """Download PDFs from the extracted links."""
    for pdf_url in pdf_links:
        filename = os.path.join(DOWNLOAD_FOLDER, pdf_url.split("/")[-1])
        
        print(f"Downloading {pdf_url}...")
        response = requests.get(pdf_url, stream=True)
        
        if response.status_code == 200:
            with open(filename, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print(f"Saved: {filename}")
        else:
            print(f"Failed to download: {pdf_url}")

if __name__ == "__main__":
    pdf_links = get_pdf_links(JFK_URL)
    if pdf_links:
        print(f"Found {len(pdf_links)} PDFs. Downloading...")
        download_pdfs(pdf_links)
    else:
        print("No PDFs found.")
