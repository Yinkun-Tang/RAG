import requests
from bs4 import BeautifulSoup
import json
import re

GAME_URLS = [
    "https://en.wikipedia.org/wiki/Hitman:_Codename_47",
    "https://en.wikipedia.org/wiki/Hitman_2:_Silent_Assassin",
    "https://en.wikipedia.org/wiki/Hitman:_Contracts",
    "https://en.wikipedia.org/wiki/Hitman:_Blood_Money",
    "https://en.wikipedia.org/wiki/Hitman:_Absolution",
    "https://en.wikipedia.org/wiki/Hitman_(2016_video_game)",
    "https://en.wikipedia.org/wiki/Hitman_2_(2018_video_game)",
    "https://en.wikipedia.org/wiki/Hitman_3"
]

OUTPUT_FILE = "../data/raw/hitman_games.json"

def clean_text(element):
    for sup in element.find_all("sup", class_="reference"):
        sup.decompose()
    text = element.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text

def parse_game_page(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/143.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    
    content_div = soup.find("div", id="mw-content-text")
    if not content_div:
        return []
    
    game_title = soup.find("h1", id="firstHeading").text.strip()
    sections_data = []
    current_section = None
    
    for element in content_div.find_all(["h2", "h3", "p", "ul", "ol"], recursive=True):
        if element.name in ["h2", "h3"]:
            section_title = element.get_text(separator=" ", strip=True)
            section_title = re.sub(r"\[edit\]$", "", section_title).strip()
            current_section = section_title
        elif element.name in ["p", "ul", "ol"]:
            text = clean_text(element)
            if text:
                sections_data.append({
                    "page_title": game_title,
                    "section": current_section,
                    "subsection": None,
                    "text": text,
                    "url": url
                })
    return sections_data

def main():
    all_data = []
    for url in GAME_URLS:
        game_data = parse_game_page(url)
        all_data.extend(game_data)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
