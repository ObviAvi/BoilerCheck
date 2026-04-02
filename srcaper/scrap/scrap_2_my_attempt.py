import requests
from bs4 import BeautifulSoup
import json


def scrape_policy_page_final(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Structure to hold the scraped data
    policy_data = {
        "document_id": url.split('/')[-2], # Extracts 'ia4' from the URL
        "title": soup.find('h1').get_text(strip=True) if soup.find('h1') else "Unknown Title",
        "domain": "purdue.edu",
        "url": url,
        "effective_date": "", 
        "last_revised": "", 
        "sections": []
    }
    # Might need to change logic based on if the pages have different HTML structure
    
    content_div = soup.find('div', class_= 'content')
    firstp = content_div.find('p')
    raw_text = firstp.get_text(separator='\n', strip=True) if firstp else ""

    for line in raw_text.split('\n'):
        if line.startswith("Date Issued"):
            policy_data["effective_date"] = line.replace("Date Issued:", "").strip()
        if line.startswith("Date Last Revised"):
            policy_data["last_revised"] = line.replace("Date Last Revised:", "").strip()


    for header in soup.find_all('h2'):
        section_title = header.get_text(strip=True)
        section_text = []
        
        # Finding all sibling elements after header until the next header
        for sibling in header.find_next_siblings():
            if sibling.name == 'h2': # Stops when hit we hit next section
                break
            if sibling.name in ['p', 'ul', 'ol']: # Grab contents
                section_text.append(sibling.get_text(separator=' ', strip=True))
                
        # Append the built section to our dictionary's list
        if section_text:
            policy_data["sections"].append({
                "section_title": section_title,
                "text": "\n".join(section_text)
            })
            
    return policy_data

