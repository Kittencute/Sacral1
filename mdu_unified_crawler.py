"""
MDU Course and Program Crawler
See README.md for usage instructions and documentation.

Author: henfal -- falkis@gmail.com
"""
import requests
import time
import random
import json
import logging
from typing import Optional, Dict, Any, List, Literal
from bs4 import BeautifulSoup
from datetime import datetime
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import argparse

class UnifiedMDUCrawler:
    def __init__(
        self,
        start_id: int,
        end_id: int,
        crawl_type: Literal['course', 'program'],
        output_dir: str = "mdu_data_url",
        min_delay: float = 2.0,
        max_delay: float = 5.0,
        verbose: bool = False,
        no_delay: bool = False
    ):
        self.base_urls = {
            'course': "https://www.mdu.se/utbildning/kursplan",
            'program': "https://www.mdu.se/utbildning/utbildningsplan"
        }
        self.crawl_type = crawl_type
        self.base_url = self.base_urls[crawl_type]
        self.start_id = start_id
        self.end_id = end_id
        self.output_dir = Path(output_dir) / crawl_type
        self.html_dir = self.output_dir / "html"
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.verbose = verbose
        self.no_delay = no_delay
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.html_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        self.setup_logging()
        self.items_by_code = defaultdict(lambda: {"giltig_fran": datetime.min})

    def setup_logging(self):
        log_file = self.output_dir / 'crawler.log'
        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def rotate_user_agent(self):
        new_agent = random.choice(self.user_agents)
        self.session.headers.update({'User-Agent': new_agent})
        self.logger.debug(f"Rotated user agent to: {new_agent}")

    def smart_delay(self):
        if not self.no_delay:
            delay = random.uniform(self.min_delay, self.max_delay)
            if self.verbose:
                self.logger.info(f"Waiting {delay:.2f} seconds")
            time.sleep(delay)

    def fetch_page(self, item_id: int) -> Optional[str]:
        url = f"{self.base_url}?id={item_id}"
        self.logger.info(f"Fetching {self.crawl_type} ID: {item_id}")
        try:
            self.rotate_user_agent()
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            if "$details.name" not in response.text:
                return response.text
            return None
        except Exception as e:
            self.logger.error(f"Error fetching {self.crawl_type} ID {item_id}: {str(e)}")
            return None

    def save_html(self, item_id: int, content: str):
        file_path = self.html_dir / f"{item_id}.html"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        self.logger.info(f"Saved HTML for {self.crawl_type} ID {item_id}")

    def extract_date(self, date_str: str) -> Optional[datetime]:
        try:
            return datetime.strptime(date_str.strip(), "%Y-%m-%d")
        except ValueError:
            import re
            match = re.match(r"(Hösttermin|Vårtermin) (\d{4})", date_str)
            if match:
                year = int(match.group(2))
                month = 8 if match.group(1) == "Hösttermin" else 1
                return datetime(year, month, 1)
        return None

    def detect_languages(self, text: str) -> List[str]:
        text = text.lower()
        languages = set()
        
        swedish_indicators = [
            "huvudsakliga undervisningsspråket är svenska",
            "undervisning sker på svenska",
            "undervisningen sker på svenska",
            "undervisningen genomförs på svenska",
            "programmet ges på svenska",
            "programmet genomförs på svenska",
            "examination sker på svenska"
        ]
        
        english_indicators = [
            "huvudsakliga undervisningsspråket är engelska",
            "undervisning sker på engelska",
            "undervisningen sker på engelska",
            "undervisningen genomförs på engelska",
            "programmet ges på engelska",
            "programmet genomförs på engelska",
            "examination sker på engelska",
            "kurslitteraturen är på engelska",
            "litteraturen är på engelska"
        ]
        
        for indicator in swedish_indicators:
            if indicator in text:
                languages.add("svenska")
                break
        
        for indicator in english_indicators:
            if indicator in text:
                languages.add("engelska")
                break
        
        if not languages:
            if "svenska" in text:
                languages.add("svenska")
            if "engelska" in text:
                languages.add("engelska")
        
        return sorted(list(languages))

    def extract_course_info(self, html_content: str, course_id: int) -> Dict[str, Any]:
        soup = BeautifulSoup(html_content, 'html.parser')
        course_data = {}
        name = soup.find('h1', class_="mdh-header-break-word")
        if name is not None:
            course_data['name'] = name.text[11:]
        
        inactive_text = soup.find(text="Denna kursplan är inte aktuell och ges inte längre")
        course_data['is_active'] = False if inactive_text else True
        
        details_block = soup.find('div', class_='mdh-details-block')
        if details_block:
            detail_items = details_block.find_all('div', class_='mdh-details-block__item')
            for item in detail_items:
                header = item.find('div', class_='mdh-details-block__header')
                content = item.find('div', class_='mdh-details-block__content')
                if header and content and "visa tidigare/senare versioner" not in header.get_text(strip=True).lower():
                    key = header.get_text(strip=True).lower()
                    value = content.get_text(strip=True)
                    course_data[key] = value

        text_sections = soup.find_all('div', class_='mdh-text-section')
        for section in text_sections:
            header = section.find(['h2'])
            if header:
                key = header.get_text(strip=True).lower()
                if key == "examination":
                    paragraphs = section.find_all('p')
                    if paragraphs:
                        course_data["examination"] = paragraphs[0].get_text(" ", strip=True)
                        continue
                if key == "innehåll":
                    paragraphs = section.find_all('p')
                    if paragraphs:
                        course_data["innehåll"] = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
                        continue
                if key not in {"betyg", "undervisning", "litteraturlistor"}:
                    content = []
                    for sibling in header.next_siblings:
                        if sibling.name == 'h2':
                            break
                        if sibling.string and sibling.string.strip():
                            content.append(sibling.string.strip())
                    if content:
                        course_data[key] = " ".join(content)
        
        course_data['source_id'] = str(course_id)
        return course_data

    def extract_program_info(self, html_content: str, program_id: int) -> Dict[str, Any]:
        soup = BeautifulSoup(html_content, 'html.parser')
        program_data = {}

        title_element = soup.find('title')
        if title_element:
            title_text = title_element.text.strip()
            if "$details.name" in title_text:
                return {'source_id': str(program_id), 'is_valid': False}
            
        if "Utbildningsplan -" in title_text:
            program_name = title_text.split("Utbildningsplan -")[1].strip()
            program_name = program_name.replace("- Mälardalens Universitet", "").replace("- Mälardalens universitet", "").strip()
            program_data['name'] = program_name
        else:
            program_data['name'] = title_text.replace("- Mälardalens Universitet", "").replace("- Mälardalens universitet", "").strip()

        inactive_text = soup.find(text="Denna utbildningsplan är inte aktuell och ges inte längre")
        program_data['is_active'] = False if inactive_text else True

        details_block = soup.find('div', class_='mdh-details-block')
        if details_block:
            for item in details_block.find_all('div', class_='mdh-details-block__item'):
                header = item.find('div', class_='mdh-details-block__header')
                content = item.find('div', class_='mdh-details-block__content')
                if header and content:
                    key = header.get_text(strip=True).lower()
                    if "version" not in key.lower():
                        value = content.get_text(strip=True)
                        program_data[key] = value

        goal_sections = {
            'kunskap och förståelse': [],
            'färdighet och förmåga': [],
            'värderingsförmåga och förhållningssätt': []
        }
        text_sections = soup.find_all('div', class_='mdh-text-section')
        current_year = None
        year_contents = defaultdict(list)
        found_language_section = False
        
        for section in text_sections:
            header = section.find(['h2', 'h3'])
            if not header:
                continue
            header_text = header.get_text(strip=True).lower()
            if header_text == "innehåll":
                paragraphs = section.find_all('p')
                if paragraphs:
                    program_data["innehåll"] = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
                    continue
            if "årskurs" in header_text:
                current_year = header_text
                continue
            content = []
            for sibling in header.next_siblings:
                if sibling.name in ['h2', 'h3']:
                    break
                if sibling.string and sibling.string.strip():
                    content.append(sibling.string.strip())
            if content:
                content_text = " ".join(content)
                if header_text == "undervisningsspråk":
                    found_language_section = True
                    program_data['undervisningsspråk'] = self.detect_languages(content_text)
                elif current_year:
                    year_contents[current_year].append(content_text)
                elif header_text in goal_sections:
                    goal_sections[header_text].append(content_text)
                else:
                    program_data[header_text] = content_text

        if not found_language_section:
            program_data['undervisningsspråk'] = []

        for goal_type, contents in goal_sections.items():
            if contents:
                program_data[goal_type] = " ".join(contents)
        if year_contents:
            program_data['årskurser'] = dict(year_contents)
        program_data['source_id'] = str(program_id)
        return program_data

    def crawl(self):
        self.logger.info(f"Starting {self.crawl_type} crawl from ID {self.start_id} to {self.end_id}")
        self.logger.info(f"Delay between requests: {'Disabled' if self.no_delay else f'{self.min_delay}-{self.max_delay} seconds'}")
        
        items_path = self.output_dir / f'{self.crawl_type}s.jsonl'
        newest_versions_path = self.output_dir / 'newest_versions.jsonl'
        
        def datetime_handler(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

        self.items_by_code.clear()

        with open(items_path, 'w', encoding='utf-8') as items_file:
            for item_id in tqdm(range(self.start_id, self.end_id + 1)):
                content = self.fetch_page(item_id)
                if content:
                    self.save_html(item_id, content)
                    
                    if self.crawl_type == 'course':
                        item_data = self.extract_course_info(content, item_id)
                    else:
                        item_data = self.extract_program_info(content, item_id)

                    if not item_data.get('is_valid', True):
                        continue

                    item_data["url"] = f"{self.base_url}?id={item_id}"

                    identifier_key = 'kurskod' if self.crawl_type == 'course' else 'programkod'
                    date_key = 'giltig från'
                    
                    code = item_data.get(identifier_key)
                    date_str = item_data.get(date_key)
                    
                    if code and date_str:
                        date = self.extract_date(date_str)
                        if date:
                            current_data = {
                                'code': code,
                                'title': item_data.get('title', ''),
                                'name': item_data.get('name', '') if self.crawl_type == 'program' else '',
                                'giltig_fran': date,
                                'id': item_id,
                                'is_active': item_data.get('is_active', True)
                            }
                            
                            if date > self.items_by_code[code].get('giltig_fran', datetime.min):
                                self.items_by_code[code] = current_data
                                self.logger.info(f"Found newer version of {code}: {date}")
                    
                    json.dump(item_data, items_file, ensure_ascii=False, default=datetime_handler)
                    items_file.write('\n')
                
                self.smart_delay()

        with open(newest_versions_path, 'w', encoding='utf-8') as f:
            for code, item_data in self.items_by_code.items():
                if code:
                    json.dump({
                        'code': code,
                        'title': item_data.get('title', ''),
                        'name': item_data.get('name', ''),
                        'giltig_fran': item_data['giltig_fran'].isoformat() if item_data.get('giltig_fran') else None,
                        'id': item_data.get('id'),
                        'is_active': item_data.get('is_active', True)
                    }, f, ensure_ascii=False)
                    f.write('\n')

        self.logger.info(f"Crawling completed. Results saved to:")
        self.logger.info(f"- {self.crawl_type.title()} data: {items_path}")
        self.logger.info(f"- Newest versions: {newest_versions_path}")
        self.logger.info(f"- HTML files: {self.html_dir}")

def main():
    parser = argparse.ArgumentParser(description='Unified MDU Course and Program Crawler')
    parser.add_argument('--course-range', nargs=2, type=int, metavar=('START', 'END'),
                      help='Start and end IDs for course crawling (e.g., --course-range 25000 35000)')
    parser.add_argument('--program-range', nargs=2, type=int, metavar=('START', 'END'),
                      help='Start and end IDs for program crawling (e.g., --program-range 200 2000)')
    parser.add_argument('--output-dir', type=str, default='mdu_data_url',
                      help='Output directory for all data (default: mdu_data_url)')
    parser.add_argument('--min-delay', type=float, default=2.0,
                      help='Minimum delay between requests (default: 2.0)')
    parser.add_argument('--max-delay', type=float, default=5.0,
                      help='Maximum delay between requests (default: 5.0)')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    parser.add_argument('--no-delay', action='store_true',
                      help='Disable delay between requests')

    args = parser.parse_args()

    if not args.course_range and not args.program_range:
        parser.error("At least one of --course-range or --program-range must be specified")

    if args.course_range:
        print(f"\nStarting crawl for courses from ID {args.course_range[0]} to {args.course_range[1]}")
        course_crawler = UnifiedMDUCrawler(
            start_id=args.course_range[0],
            end_id=args.course_range[1],
            crawl_type='course',
            output_dir=args.output_dir,
            min_delay=args.min_delay,
            max_delay=args.max_delay,
            verbose=args.verbose,
            no_delay=args.no_delay
        )
        course_crawler.crawl()

    if args.program_range:
        print(f"\nStarting crawl for programs from ID {args.program_range[0]} to {args.program_range[1]}")
        program_crawler = UnifiedMDUCrawler(
            start_id=args.program_range[0],
            end_id=args.program_range[1],
            crawl_type='program',
            output_dir=args.output_dir,
            min_delay=args.min_delay,
            max_delay=args.max_delay,
            verbose=args.verbose,
            no_delay=args.no_delay
        )
        program_crawler.crawl()


if __name__ == "__main__":
    main()
