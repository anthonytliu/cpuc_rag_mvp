# 📁 california-reg-rag (Enhanced version with diff check for new PDFs)
# Scrape all documents and metadata for CPUC Proceeding R2207005

import hashlib
import json
import os
import re
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait

# Setup
PROCEEDING_LIST = ["R2207005"]
BASE_URL = "https://apps.cpuc.ca.gov/apex/f?p=401:1:0"
DOWNLOAD_DIR = os.path.abspath("./cpuc_csvs")

chrome_options = Options()
# chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": DOWNLOAD_DIR,
    "download.prompt_for_download": False,
    "plugins.always_open_pdf_externally": True
})


def sanitize_filename(filename_to_sanitize):
    """Sanitize filename to remove invalid characters"""
    return re.sub(r'[<>:"/\\|?*]', '_', filename_to_sanitize)


def wait_for_download(download_dir, timeout=30):
    """Wait for download to complete"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        files = os.listdir(download_dir)
        if any(file.endswith('.crdownload') for file in files):
            time.sleep(1)
            continue
        return True
    return False


def create_url_hash(url):
    """Create a hash of the URL for unique identification"""
    return hashlib.md5(url.encode()).hexdigest()


def load_download_history(proceeding_num):
    """Load the history of downloaded files"""
    history_file_path = os.path.join(DOWNLOAD_DIR, f"{proceeding_num.lower()}_download_history.json")
    if os.path.exists(history_file_path):
        with open(history_file_path, 'r') as file_handle:
            return json.load(file_handle)
    return {}


def save_download_history(proceeding_num, history_data):
    """Save the history of downloaded files"""
    history_file_path = os.path.join(DOWNLOAD_DIR, f"{proceeding_num.lower()}_download_history.json")
    with open(history_file_path, 'w') as file_handle:
        json.dump(history_data, file_handle, indent=2)


def extract_pdf_urls_from_csv(csv_file_path):
    """Extract PDF URLs from CSV file"""
    pdf_url_list = []
    try:
        dataframe = pd.read_csv(csv_file_path)
        print(f"📊 Processing CSV with {len(dataframe)} rows...")

        for row_index, row_data in dataframe.iterrows():
            doc_type_cell_content = row_data.get("Document Type")
            if isinstance(doc_type_cell_content, str) and "<a href=" in doc_type_cell_content:
                # Parse the HTML to extract the actual URL
                try:
                    cell_soup = BeautifulSoup(doc_type_cell_content, "html.parser")
                    link_element = cell_soup.find("a")
                    if link_element and link_element.get("href"):
                        document_url = link_element["href"]
                        pdf_url_list.append(document_url)
                except Exception as parse_error:
                    print(f"  ⚠️ Error parsing HTML in cell: {parse_error}")

    except Exception as csv_error:
        print(f"⚠️ Error reading CSV: {csv_error}")

    return pdf_url_list


def compare_and_find_new_urls(current_url_list, download_history_data):
    """Compare current URLs with download history and find new ones"""
    new_url_list = []
    updated_url_list = []

    for current_url in current_url_list:
        url_hash_value = create_url_hash(current_url)

        if url_hash_value not in download_history_data:
            # Completely new URL
            new_url_list.append(current_url)
        else:
            # URL exists, check if it needs to be re-downloaded
            # You could add additional checks here, like date comparison
            # For now, we'll assume existing URLs don't need re-download
            pass

    return new_url_list, updated_url_list


def update_download_history(proceeding_num, document_url, file_name, download_status="downloaded"):
    """Update download history with new entry"""
    history_data = load_download_history(proceeding_num)
    url_hash_value = create_url_hash(document_url)

    history_data[url_hash_value] = {
        "url": document_url,
        "filename": file_name,
        "status": download_status,
        "download_date": datetime.now().isoformat(),
        "last_checked": datetime.now().isoformat()
    }

    save_download_history(proceeding_num, history_data)


# Add this new function definition after your existing helper functions
# (e.g., after update_download_history)

def download_pdfs_from_url(page_url, proceeding_num, pdf_dir):
    """Visits a URL, finds all PDF links, and downloads unique ones."""
    print(f"  -> Visiting search result: {page_url}")
    new_downloads_count = 0
    try:
        page_response = requests.get(page_url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        page_response.raise_for_status()
        soup = BeautifulSoup(page_response.text, 'html.parser')

        history = load_download_history(proceeding_num)

        # Find all links that end with .pdf
        pdf_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].lower().endswith('.pdf')]

        for link in pdf_links:
            # Construct the full URL if it's relative
            if link.startswith('/'):
                pdf_url = "https://www.cpuc.ca.gov" + link
            else:
                pdf_url = link

            # Use the URL itself as a unique identifier to check against history
            url_hash = create_url_hash(pdf_url)
            if url_hash not in history:
                filename = sanitize_filename(os.path.basename(pdf_url))
                pdf_path = os.path.join(pdf_dir, filename)

                print(f"    Found new PDF: {filename}")
                try:
                    pdf_response = requests.get(pdf_url, stream=True, timeout=60)
                    pdf_response.raise_for_status()

                    with open(pdf_path, 'wb') as f:
                        for chunk in pdf_response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    print(f"    ✅ Downloaded: {filename}")
                    # Use the full PDF URL for the history entry
                    update_download_history(proceeding_num, pdf_url, filename, "downloaded_from_google_search")
                    new_downloads_count += 1
                    time.sleep(0.5)  # Be respectful

                except Exception as download_error:
                    print(f"    ⚠️ Failed to download {pdf_url}: {download_error}")
                    update_download_history(proceeding_num, pdf_url, filename, "error_google_search")
            else:
                print(f"    Skipping already downloaded PDF: {history[url_hash]['filename']}")

    except Exception as e:
        print(f"  ⚠️ Could not process page {page_url}: {e}")

    return new_downloads_count


print("🚀 Launching Chrome...")
driver = webdriver.Chrome(options=chrome_options)
driver.get(BASE_URL)
wait = WebDriverWait(driver, 10)

for PROCEEDING in PROCEEDING_LIST:
    PDF_DIR = os.path.abspath(f"./cpuc_pdfs/{PROCEEDING}")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)

    # Load download history
    download_history = load_download_history(PROCEEDING)
    print(f"📚 Loaded download history with {len(download_history)} entries")

    try:
        print(f"🔍 Searching for proceeding {PROCEEDING}...")
        input_box = wait.until(ec.presence_of_element_located((By.ID, "P1_PROCEEDING_NUM")))
        input_box.clear()
        input_box.send_keys(PROCEEDING)
        driver.find_element(By.ID, "P1_SEARCH").click()

        print("➡️ Clicking result link...")
        wait.until(ec.presence_of_element_located(
            (By.XPATH, f"//td[@headers='PROCEEDING_STATUS_DESC']/a[contains(@href, '{PROCEEDING}')]"))).click()

        # Extract metadata
        print("📝 Extracting metadata...")
        soup = BeautifulSoup(driver.page_source, "html.parser")
        metadata = {}
        table = soup.find("table", class_="t14Standard")
        if table:
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) == 2:
                    key = cells[0].text.strip().replace(":", "")
                    val = cells[1].text.strip()
                    metadata[key] = val

        with open(os.path.join(PDF_DIR, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Metadata saved to {PDF_DIR}/metadata.json")

        print("📂 Navigating to Documents tab...")
        wait.until(ec.presence_of_element_located((By.LINK_TEXT, "Documents"))).click()
        time.sleep(2)

        # Download CSV
        print("📥 Starting CSV download process...")
        csv_files_downloaded = []

        try:
            download_btn = wait.until(ec.presence_of_element_located((By.XPATH, "//input[@value='Download']")))
            download_btn.click()

            # Wait for download to complete
            wait_for_download(DOWNLOAD_DIR)
            time.sleep(2)

            # Find the most recent CSV file
            csv_files = sorted(
                [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".csv")],
                key=lambda x: os.path.getmtime(os.path.join(DOWNLOAD_DIR, x)),
                reverse=True
            )

            if csv_files:
                latest_csv = csv_files[0]
                new_csv_name = f"{PROCEEDING.lower()}_resultCSV.csv"
                old_path = os.path.join(DOWNLOAD_DIR, latest_csv)
                new_path = os.path.join(DOWNLOAD_DIR, new_csv_name)

                # Rename the file
                os.rename(old_path, new_path)
                csv_files_downloaded.append(new_csv_name)
                print(f"✅ Downloaded and renamed to: {new_csv_name}")

        except TimeoutException:
            print("⚠️ Download button not found, might be on last page")
            # Check if we have a previous CSV file to work with
            expected_csv = f"{PROCEEDING.lower()}_resultCSV.csv"
            if os.path.exists(os.path.join(DOWNLOAD_DIR, expected_csv)):
                csv_files_downloaded.append(expected_csv)
                print(f"📄 Using existing CSV file: {expected_csv}")

        # Process CSV files and perform diff check
        print("🔄 Performing diff check for new PDFs...")

        all_current_urls = []
        for csv_file in csv_files_downloaded:
            csv_path = os.path.join(DOWNLOAD_DIR, csv_file)
            urls = extract_pdf_urls_from_csv(csv_path)
            all_current_urls.extend(urls)

        print(f"🔗 Found {len(all_current_urls)} total document URLs")

        # Compare with download history
        new_urls, updated_urls = compare_and_find_new_urls(all_current_urls, download_history)

        print(f"🆕 Found {len(new_urls)} new URLs to download")
        if updated_urls:
            print(f"🔄 Found {len(updated_urls)} updated URLs to re-download")

        # Download only new PDFs
        urls_to_download = new_urls + updated_urls

        if not urls_to_download:
            print("✅ No new PDFs to download. All files are up to date!")
        else:
            print(f"📥 Starting download of {len(urls_to_download)} new/updated PDFs...")

            for i, doc_url in enumerate(urls_to_download, 1):
                print(f"🔗 Processing document {i}/{len(urls_to_download)}: {doc_url}")

                try:
                    doc_page = requests.get(doc_url, timeout=30)
                    doc_soup = BeautifulSoup(doc_page.text, "html.parser")

                    # Get the title for filename
                    title_td = doc_soup.find("td", class_="ResultTitleTD")
                    if title_td:
                        title_text = title_td.get_text(strip=True)
                        # Extract just the document ID and description (before "Proceeding:")
                        title_parts = title_text.split("Proceeding:")
                        if title_parts:
                            title = title_parts[0].strip()
                            # Remove line breaks and extra spaces
                            title = re.sub(r'\s+', ' ', title)
                            filename = sanitize_filename(title)
                        else:
                            filename = f"document_{i}"
                    else:
                        filename = f"document_{i}"

                    # Find PDF download link
                    result_td = doc_soup.find("td", class_="ResultLinkTD")
                    if result_td:
                        pdf_link = result_td.find("a", string="PDF")
                        if pdf_link and pdf_link.get("href"):
                            pdf_url = "https://docs.cpuc.ca.gov" + pdf_link["href"]
                            pdf_path = os.path.join(PDF_DIR, f"{filename}.pdf")

                            print(f"⬇️ Downloading: {filename}.pdf")

                            pdf_response = requests.get(pdf_url, stream=True, timeout=60)
                            pdf_response.raise_for_status()

                            with open(pdf_path, 'wb') as f:
                                for chunk in pdf_response.iter_content(chunk_size=8192):
                                    f.write(chunk)

                            print(f"✅ Downloaded: {filename}.pdf")

                            # Update download history
                            update_download_history(PROCEEDING, doc_url, f"{filename}.pdf", "downloaded")
                        else:
                            print(f"⚠️ No PDF link found for {doc_url}")
                            update_download_history(PROCEEDING, doc_url, f"{filename}.pdf", "no_pdf_link")
                    else:
                        print(f"⚠️ No result link section found for {doc_url}")
                        update_download_history(PROCEEDING, doc_url, f"{filename}.pdf", "no_result_section")

                except Exception as e:
                    print(f"⚠️ Error processing {doc_url}: {e}")
                    update_download_history(PROCEEDING, doc_url, f"error_{i}.pdf", "error")

                # Small delay to be respectful
                time.sleep(0.5)

        # Update history for all URLs we've seen (mark as checked)
        final_history = load_download_history(PROCEEDING)
        for url in all_current_urls:
            url_hash = create_url_hash(url)
            if url_hash in final_history:
                final_history[url_hash]["last_checked"] = datetime.now().isoformat()
        save_download_history(PROCEEDING, final_history)

        print(f"📊 Final Summary:")
        print(f"  📁 Total URLs in current CSV: {len(all_current_urls)}")
        print(f"  📚 URLs in download history: {len(download_history)}")
        print(f"  🆕 New URLs downloaded: {len(new_urls)}")
        print(f"  📄 CSV files processed: {len(csv_files_downloaded)}")

        # ### NEW FEATURE: Google Search for supplementary documents ###
        print("\n" + "=" * 50)
        print(f"🔎 Starting Google Search for supplementary documents related to {PROCEEDING}...")

        # Format proceeding number for search (e.g., R2207005 -> "R.22-07-005")
        search_term_formatted = f"R.{PROCEEDING[1:3]}-{PROCEEDING[3:5]}-{PROCEEDING[5:]}"
        query = f'"{search_term_formatted}" site:cpuc.ca.gov'
        print(f"  -> Using search query: {query}")

        google_results_to_check = []
        try:
            # Use tld="com" for google.com, num=5 to get top 5 results
            for url in search(query):
                if "cpuc.ca.gov" in url:
                    google_results_to_check.append(url)
        except Exception as search_error:
            print(f"  ⚠️ Google search failed: {search_error}")

        # Process the top 3 valid results
        if not google_results_to_check:
            print("  -> No relevant cpuc.ca.gov links found in top Google results.")
        else:
            print(f"  -> Found {len(google_results_to_check)} relevant links. Checking top 3 for unique PDFs...")
            total_new_from_google = 0
            for page_url in google_results_to_check[:3]:
                total_new_from_google += download_pdfs_from_url(page_url, PROCEEDING, PDF_DIR)

            print(f"  ✅ Google search supplementary download complete. Found {total_new_from_google} new PDFs.")
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        driver.quit()
        print("🏁 Done.")
