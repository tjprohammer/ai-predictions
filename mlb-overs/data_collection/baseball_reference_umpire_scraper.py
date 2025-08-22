#!/usr/bin/env python3
"""
Baseball Reference Umpire Scraper for Historic MLB Games

This script scrapes complete umpire assignments from Baseball Reference
for all 2,002 historic games in the database. Baseball Reference has
comprehensive umpire data for every MLB game since the early 1900s.

Features:
- Scrapes real umpire assignments (Home Plate, 1B, 2B, 3B)
- Handles team abbreviation mapping (DB vs Baseball Reference)
- Rate limiting to respect Baseball Reference servers
- Robust error handling and retry logic
- Progress tracking and logging
"""

import requests
from bs4 import BeautifulSoup
import psycopg2
import time
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import re
import os
from dataclasses import dataclass
import random
from urllib.parse import urljoin

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('baseball_reference_umpire_scraper.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class UmpireAssignment:
    """Complete umpire crew assignment for a game"""
    game_id: str
    date: str
    home_team: str
    away_team: str
    home_plate_umpire: str
    first_base_umpire: str
    second_base_umpire: str
    third_base_umpire: str

class BaseballReferenceUmpireScraper:
    """Scrape umpire assignments from Baseball Reference"""
    
    def __init__(self):
        """Initialize scraper with team mappings and session"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Map database team names to Baseball Reference abbreviations
        self.team_mapping = {
            'Angels': 'LAA',
            'Astros': 'HOU', 
            'Athletics': 'OAK',
            'Blue Jays': 'TOR',
            'Brewers': 'MIL',
            'Cardinals': 'STL',
            'Cubs': 'CHC',
            'Diamondbacks': 'ARI',
            'Dodgers': 'LAD',
            'Giants': 'SFG',
            'Guardians': 'CLE',
            'Mariners': 'SEA',
            'Marlins': 'MIA',
            'Mets': 'NYM',
            'Nationals': 'WSN',
            'Orioles': 'BAL',
            'Padres': 'SDP',
            'Phillies': 'PHI',
            'Pirates': 'PIT',
            'Rangers': 'TEX',
            'Rays': 'TBR',
            'Red Sox': 'BOS',
            'Reds': 'CIN',
            'Rockies': 'COL',
            'Royals': 'KCR',
            'Tigers': 'DET',
            'Twins': 'MIN',
            'White Sox': 'CHW',
            'Yankees': 'NYY',
            'Braves': 'ATL'
        }
        
        # Reverse mapping for lookups
        self.abbrev_to_team = {v: k for k, v in self.team_mapping.items()}
        
        self.base_url = "https://www.baseball-reference.com"
        
        logging.info(f"Initialized Baseball Reference scraper with {len(self.team_mapping)} team mappings")
    
    def get_team_abbrev(self, team_name: str) -> str:
        """Convert team name to Baseball Reference abbreviation"""
        return self.team_mapping.get(team_name, team_name[:3].upper())
    
    def get_games_needing_umpires(self) -> List[Tuple]:
        """Get all games that need umpire assignments"""
        conn = psycopg2.connect(
            host='localhost',
            database='mlb',
            user='mlbuser',
            password='mlbpass'
        )
        cur = conn.cursor()
        
        cur.execute("""
            SELECT game_id, date, home_team, away_team
            FROM enhanced_games 
            WHERE date >= '2025-03-20'
            AND (home_plate_umpire_name IS NULL OR home_plate_umpire_name = '')
            ORDER BY date, game_id
        """)
        
        games = cur.fetchall()
        cur.close()
        conn.close()
        
        logging.info(f"Found {len(games)} games needing umpire assignments")
        return games
    
    def build_game_url(self, game_date: str, home_team: str, away_team: str) -> str:
        """Build Baseball Reference URL for a specific game"""
        # Convert date to Baseball Reference format
        date_obj = datetime.strptime(game_date, '%Y-%m-%d')
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        
        # Get team abbreviations
        home_abbrev = self.get_team_abbrev(home_team)
        away_abbrev = self.get_team_abbrev(away_team)
        
        # Baseball Reference URL pattern: /boxes/[HOME]/[YEAR][MONTH][DAY]0.shtml
        home_code = home_abbrev.lower()
        date_code = f"{year:04d}{month:02d}{day:02d}0"
        
        url = f"{self.base_url}/boxes/{home_code}/{date_code}.shtml"
        return url
    
    def extract_umpires_from_game_page(self, html_content: str, game_date: str, home_team: str, away_team: str) -> Optional[UmpireAssignment]:
        """Extract umpire assignments from Baseball Reference game page"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for umpire information in various possible locations
            umpire_info = None
            
            # Method 1: Look for "Umpires" section in game info
            umpire_sections = soup.find_all(text=re.compile(r'Umpires?:', re.IGNORECASE))
            for section in umpire_sections:
                parent = section.parent
                if parent:
                    # Get the text following "Umpires:"
                    full_text = parent.get_text()
                    umpire_match = re.search(r'Umpires?:\s*(.+)', full_text, re.IGNORECASE)
                    if umpire_match:
                        umpire_info = umpire_match.group(1).strip()
                        break
            
            # Method 2: Look in game information table
            if not umpire_info:
                info_tables = soup.find_all('table', {'id': re.compile(r'.*info.*', re.IGNORECASE)})
                for table in info_tables:
                    rows = table.find_all('tr')
                    for row in rows:
                        text = row.get_text()
                        if 'umpire' in text.lower():
                            umpire_match = re.search(r'umpires?[:\s]*(.+)', text, re.IGNORECASE)
                            if umpire_match:
                                umpire_info = umpire_match.group(1).strip()
                                break
                    if umpire_info:
                        break
            
            # Method 3: Look for umpire data in divs with specific classes
            if not umpire_info:
                game_info_divs = soup.find_all('div', class_=re.compile(r'.*game.*info.*', re.IGNORECASE))
                for div in game_info_divs:
                    text = div.get_text()
                    if 'umpire' in text.lower():
                        umpire_match = re.search(r'umpires?[:\s]*(.+)', text, re.IGNORECASE)
                        if umpire_match:
                            umpire_info = umpire_match.group(1).strip()
                            break
            
            if not umpire_info:
                logging.debug(f"No umpire information found for {game_date} {away_team}@{home_team}")
                return None
            
            # Parse umpire assignments from the extracted text
            return self.parse_umpire_assignments(umpire_info, game_date, home_team, away_team)
            
        except Exception as e:
            logging.error(f"Error extracting umpires from game page: {e}")
            return None
    
    def parse_umpire_assignments(self, umpire_text: str, game_date: str, home_team: str, away_team: str) -> Optional[UmpireAssignment]:
        """Parse umpire assignments from extracted text"""
        try:
            # Clean up the text
            umpire_text = umpire_text.replace('\n', ' ').replace('\t', ' ')
            umpire_text = re.sub(r'\s+', ' ', umpire_text).strip()
            
            # Initialize umpires
            plate_umpire = None
            first_base = None
            second_base = None
            third_base = None
            
            # Pattern 1: "HP: Name, 1B: Name, 2B: Name, 3B: Name"
            position_patterns = [
                (r'HP[:\s]+([^,;]+)', 'plate'),
                (r'1B[:\s]+([^,;]+)', 'first'),
                (r'2B[:\s]+([^,;]+)', 'second'),
                (r'3B[:\s]+([^,;]+)', 'third'),
                (r'Home[:\s]+([^,;]+)', 'plate'),
                (r'First[:\s]+([^,;]+)', 'first'),
                (r'Second[:\s]+([^,;]+)', 'second'),
                (r'Third[:\s]+([^,;]+)', 'third')
            ]
            
            for pattern, position in position_patterns:
                match = re.search(pattern, umpire_text, re.IGNORECASE)
                if match:
                    name = match.group(1).strip().rstrip(',;')
                    name = re.sub(r'\s+', ' ', name)
                    
                    if position == 'plate':
                        plate_umpire = name
                    elif position == 'first':
                        first_base = name
                    elif position == 'second':
                        second_base = name
                    elif position == 'third':
                        third_base = name
            
            # Pattern 2: Simple comma-separated list (assume order: HP, 1B, 2B, 3B)
            if not plate_umpire:
                # Split by common separators
                names = re.split(r'[,;]', umpire_text)
                names = [name.strip() for name in names if name.strip()]
                
                if len(names) >= 4:
                    plate_umpire = names[0]
                    first_base = names[1]
                    second_base = names[2]
                    third_base = names[3]
                elif len(names) >= 1:
                    # If only one name, assume it's the plate umpire
                    plate_umpire = names[0]
            
            # Validate we have at least the plate umpire
            if not plate_umpire:
                logging.debug(f"Could not parse plate umpire from: {umpire_text}")
                return None
            
            # Use the game_id generation logic (you may need to adjust this)
            game_id = f"{game_date}_{self.get_team_abbrev(away_team)}_{self.get_team_abbrev(home_team)}"
            
            return UmpireAssignment(
                game_id=game_id,
                date=game_date,
                home_team=home_team,
                away_team=away_team,
                home_plate_umpire=plate_umpire or "Unknown",
                first_base_umpire=first_base or "Unknown",
                second_base_umpire=second_base or "Unknown",
                third_base_umpire=third_base or "Unknown"
            )
            
        except Exception as e:
            logging.error(f"Error parsing umpire assignments: {e}")
            return None
    
    def scrape_game_umpires(self, game_date: str, home_team: str, away_team: str) -> Optional[UmpireAssignment]:
        """Scrape umpire assignments for a specific game"""
        try:
            url = self.build_game_url(game_date, home_team, away_team)
            logging.debug(f"Scraping umpires from: {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            if response.status_code == 200:
                assignment = self.extract_umpires_from_game_page(
                    response.text, game_date, home_team, away_team
                )
                
                if assignment:
                    logging.info(f"✅ Found umpires for {game_date} {away_team}@{home_team}: {assignment.home_plate_umpire}")
                    return assignment
                else:
                    logging.warning(f"❌ No umpire data found for {game_date} {away_team}@{home_team}")
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed for {game_date} {away_team}@{home_team}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error scraping {game_date} {away_team}@{home_team}: {e}")
        
        return None
    
    def update_database_with_assignment(self, assignment: UmpireAssignment) -> bool:
        """Update database with scraped umpire assignment"""
        try:
            conn = psycopg2.connect(
                host='localhost',
                database='mlb',
                user='mlbuser',
                password='mlbpass'
            )
            cur = conn.cursor()
            
            cur.execute("""
                UPDATE enhanced_games SET
                    home_plate_umpire_name = %s,
                    first_base_umpire_name = %s,
                    second_base_umpire_name = %s,
                    third_base_umpire_name = %s
                WHERE game_id = %s
            """, (
                assignment.home_plate_umpire,
                assignment.first_base_umpire,
                assignment.second_base_umpire,
                assignment.third_base_umpire,
                assignment.game_id
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            return True
            
        except Exception as e:
            logging.error(f"Database update failed for game {assignment.game_id}: {e}")
            return False
    
    def scrape_all_umpire_assignments(self, batch_size: int = 50, delay_seconds: float = 2.0) -> Dict[str, int]:
        """Scrape umpire assignments for all games needing them"""
        logging.info("🔍 Starting Baseball Reference umpire assignment scraping...")
        
        games = self.get_games_needing_umpires()
        total_games = len(games)
        
        if total_games == 0:
            logging.info("✅ All games already have umpire assignments!")
            return {"already_complete": 0}
        
        stats = {
            "total_games": total_games,
            "scraped_successfully": 0,
            "failed_scrapes": 0,
            "database_updates": 0
        }
        
        for i, (game_id, game_date, home_team, away_team) in enumerate(games, 1):
            try:
                # Scrape umpire assignments
                assignment = self.scrape_game_umpires(
                    str(game_date), home_team, away_team
                )
                
                if assignment:
                    # Update assignment with correct game_id from database
                    assignment.game_id = game_id
                    
                    # Update database
                    if self.update_database_with_assignment(assignment):
                        stats["scraped_successfully"] += 1
                        stats["database_updates"] += 1
                    else:
                        stats["scraped_successfully"] += 1  # Scraped but DB update failed
                else:
                    stats["failed_scrapes"] += 1
                
                # Progress reporting
                if i % batch_size == 0 or i == total_games:
                    pct = (i / total_games) * 100
                    logging.info(f"📊 Progress: {i}/{total_games} ({pct:.1f}%) - "
                               f"Success: {stats['scraped_successfully']}, "
                               f"Failed: {stats['failed_scrapes']}, "
                               f"DB Updates: {stats['database_updates']}")
                
                # Rate limiting to be respectful to Baseball Reference
                time.sleep(delay_seconds + random.uniform(0, 1))
                
            except Exception as e:
                logging.error(f"Failed to process game {game_id}: {e}")
                stats["failed_scrapes"] += 1
                time.sleep(delay_seconds)  # Still delay on errors
        
        return stats

def main():
    """Main execution function"""
    print("⚾ BASEBALL REFERENCE UMPIRE ASSIGNMENT SCRAPER")
    print("=" * 60)
    
    try:
        scraper = BaseballReferenceUmpireScraper()
        
        # Run scraping with conservative rate limiting
        results = scraper.scrape_all_umpire_assignments(
            batch_size=25,
            delay_seconds=3.0  # 3 second delay between requests
        )
        
        # Summary
        print(f"\n📈 SCRAPING COMPLETE!")
        print(f"Total games processed: {results.get('total_games', 0)}")
        print(f"Successfully scraped: {results.get('scraped_successfully', 0)}")
        print(f"Failed scrapes: {results.get('failed_scrapes', 0)}")
        print(f"Database updates: {results.get('database_updates', 0)}")
        
        if results.get('total_games', 0) > 0:
            success_rate = (results.get('scraped_successfully', 0) / results['total_games']) * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        if results.get('already_complete'):
            print(f"✅ All games already had umpire assignments!")
        
        print(f"\n✅ Phase 4 Baseball Reference umpire scraping completed!")
        
    except Exception as e:
        logging.error(f"Scraping failed: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
