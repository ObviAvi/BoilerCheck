import requests
from bs4 import BeautifulSoup
import json


def scrape_policy_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Structure to hold the scraped data
    policy_data = {
        "document_id": url.split('/')[-2], # Extracts 'ia4' from the URL
        "title": soup.find('h1').get_text(strip=True) if soup.find('h1') else "Unknown Title",
        "domain": "purdue.edu",
        "url": url,
        "effective_date": "", # TODO need to find specific HTML tag for this
        "sections": []
    }
    # Might need to change logic based on if the pages have different HTML structure
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

# data = scrape_policy_page("https://www.purdue.edu/vpec/policies/academic-research-affairs/ia4/")
# print(json.dumps(data, indent=4))