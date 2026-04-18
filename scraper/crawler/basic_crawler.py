import requests
from collections import deque
from scrap import scrap_2_my_attempt
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time


def crawler(starter_url, max_depth=10):
    queue = deque([(starter_url)]) # Initialize queue with the starter URL
    visited = set() #creates hashtable
    pages_crawled = 0
    all_data = [] # List to hold data from all crawled pages
    
    while queue and pages_crawled < max_depth:
        url = queue.popleft()
        if url in visited:
            continue

        print(f"Crawling: {url}")

        try:
            r = requests.get(url)
            if r.status_code != 200:
                print(f"Failed to retrieve {url} (status code: {r.status_code})")
                continue
            
            visited.add(url)
            pages_crawled += 1

            soup = BeautifulSoup(r.content, 'html.parser')
            page_data = scrap_2_my_attempt.scrape_policy_page_final(url)

            for link in soup.find_all('a'):
                href = link.get('href')
                if not href:
                    continue
                
                full_url = urljoin(url, href)
                full_url = full_url.split('#')[0] # Remove any fragment identifiers
                parsed_url = urlparse(full_url)

                if parsed_url.netloc != "www.purdue.edu":
                    continue
                
                current_depth = urlparse(url).path.count('/')
                new_depth = parsed_url.path.count('/')

                if new_depth >= current_depth and full_url not in visited:
                    queue.append(full_url)


            all_data.append(page_data)
            time.sleep(0.5) # Be polite and avoid overwhelming the server

        except Exception as e:
            print(f"Error crawling {url}: {e}")
            continue
            
    with open("policies.json", "w") as f:
        json.dump(all_data, f, indent=4)
    print(f"Saved {len(all_data)} pages to policies.json")

starter_url = "https://www.housing.purdue.edu/"
crawler(starter_url, max_depth=5)  # small max_depth for testing