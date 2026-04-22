import requests
from bs4 import BeautifulSoup
import re

# Many purdue.edu subdomains sit behind Akamai/TSPD bot protection that serves
# a JS challenge to clients with no User-Agent.
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

HIGH_VALUE_PATHS = [
    "policy", "policies", "regulation", "regulations",
    "registrar", "housing", "financial-aid", "financialaid",
    "bursar", "studentregulations", "provost", "conduct",
    "handbook", "catalog", "scholarships", "scholarship",
    "tuition", "fees", "academic", "dean", "admissions",
    "appeal", "student-rights", "ombudsman",
]

TITLE_KEYWORDS = [
    "policy", "policies", "regulation", "procedure", "guidelines",
    "handbook", "requirements", "terms", "conditions", "agreement",
    "eligibility", "scholarship", "financial aid", "tuition", "fees",
    "conduct", "code", "rights", "responsibilities", "appeal",
    "catalog", "bulletin", "deadline", "compliance", "rule", "rules",
    "housing", "residential", "academic", "enrollment", "registration",
]

BODY_KEYWORDS = [
    "policy", "policies", "regulation", "procedure", "guidelines",
    "compliance", "code of conduct", "requirement", "requirements",
    "terms and conditions", "eligibility", "financial aid", "scholarship",
    "tuition", "fees", "refund", "appeal", "grievance", "violation",
    "sanction", "probation", "suspension", "dismissal", "academic standing",
    "credit hours", "enrollment", "withdrawal", "deadline", "waiver",
    "housing contract", "residential", "meal plan", "student account",
    "billing", "payment plan", "disbursement", "award letter",
    "satisfactory academic progress", "handbook", "bulletin",
]

SCORE_THRESHOLD = 10


def is_definitely_useless(url: str, title: str, word_count: int, has_structure: bool) -> bool:
    """Short page with no tables/lists — likely a stub, nav, or announcement."""
    return word_count < 150 and not has_structure


def score_page(url: str, title: str, text: str, word_count: int, has_structure: bool) -> int:
    """
    Keyword-weighted relevance score. Keep if >= SCORE_THRESHOLD.
    No negative penalties — 0 means uncertain, not irrelevant.
    """
    score = 0
    url_lower   = url.lower()
    title_lower = title.lower()
    text_lower  = text.lower()

    # URL path — strongest pre-content signal; only count once so it can't dominate.
    for seg in HIGH_VALUE_PATHS:
        if seg in url_lower:
            score += 15
            break

    for kw in TITLE_KEYWORDS:
        if kw in title_lower:
            score += 12

    # Cap body hits per keyword to avoid runaway scores on repetitive pages.
    for kw in BODY_KEYWORDS:
        hits = text_lower.count(kw)
        if hits > 0:
            score += min(hits * 3, 9)

    if word_count >= 500:
        score += 8
    elif word_count >= 200:
        score += 4

    if has_structure:
        score += 8

    if "purdue" in title_lower and any(kw in title_lower for kw in TITLE_KEYWORDS):
        score += 15

    return score


def scrape_policy_page_final(url: str) -> dict:
    response = requests.get(url, timeout=10, headers=REQUEST_HEADERS)
    soup = BeautifulSoup(response.content, "html.parser")

    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
    elif soup.title:
        title = soup.title.get_text(strip=True)
    else:
        title = "Unknown Title"

    effective_date = ""
    last_revised   = ""
    for tag in soup.find_all(["p", "li", "span", "td"]):
        line = tag.get_text(strip=True)
        if not effective_date and re.search(r"date\s+issued", line, re.I):
            effective_date = re.sub(r"date\s+issued[:\-]?\s*", "", line, flags=re.I).strip()
        if not last_revised and re.search(r"date\s+last\s+revised|last\s+updated|last\s+modified", line, re.I):
            last_revised = re.sub(r"(date\s+last\s+revised|last\s+updated|last\s+modified)[:\-]?\s*", "", line, flags=re.I).strip()

    sections = []
    all_section_text = []

    for header in soup.find_all(["h2", "h3"]):
        section_title = header.get_text(strip=True)
        section_text  = []

        for sibling in header.find_next_siblings():
            if sibling.name in ["h2", "h3"]:
                break
            if sibling.name in ["p", "ul", "ol", "table", "div"]:
                text_chunk = sibling.get_text(separator=" ", strip=True)
                if text_chunk:
                    section_text.append(text_chunk)

        if section_text:
            joined = "\n".join(section_text)
            sections.append({
                "section_title": section_title,
                "text": joined
            })
            all_section_text.append(joined)

    # Fallback for pages with no h2/h3 structure.
    if not sections:
        body = soup.find("main") or soup.find("article") or soup.find("body")
        if body:
            raw = body.get_text(separator=" ", strip=True)
            all_section_text.append(raw)
            sections.append({"section_title": "Full Page", "text": raw})

    full_text  = " ".join(all_section_text)
    word_count = len(full_text.split())
    has_structure = bool(soup.find("table") or soup.find("ul") or soup.find("ol"))

    if is_definitely_useless(url, title, word_count, has_structure):
        final_score = 0
    else:
        final_score = score_page(url, title, full_text, word_count, has_structure)

    return {
        "document_id":    url.split("/")[-2] or url.split("/")[-1],
        "title":          title,
        "domain":         "purdue.edu",
        "url":            url,
        "effective_date": effective_date,
        "last_revised":   last_revised,
        "sections":       sections,
        "word_count":     word_count,
        "has_structure":  has_structure,
        "score":          final_score,
        "relevant":       final_score >= SCORE_THRESHOLD,
    }