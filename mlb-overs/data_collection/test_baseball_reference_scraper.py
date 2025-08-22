#!/usr/bin/env python3
"""
Test Baseball Reference Umpire Scraper

This script tests the Baseball Reference scraper with a few sample games
to verify it's working correctly before running on all 2,002 games.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from baseball_reference_umpire_scraper import BaseballReferenceUmpireScraper
import logging

def test_scraper():
    """Test the scraper with a few sample games"""
    print("🧪 TESTING BASEBALL REFERENCE UMPIRE SCRAPER")
    print("=" * 60)
    
    scraper = BaseballReferenceUmpireScraper()
    
    # Get a few recent games to test
    test_games = [
        ("2025-08-20", "Yankees", "Dodgers"),
        ("2025-08-21", "Red Sox", "Giants"),
        ("2025-08-19", "Astros", "Cubs")
    ]
    
    successful_tests = 0
    total_tests = len(test_games)
    
    for game_date, home_team, away_team in test_games:
        print(f"\n🔍 Testing: {game_date} {away_team} @ {home_team}")
        print("-" * 50)
        
        try:
            # Build URL
            url = scraper.build_game_url(game_date, home_team, away_team)
            print(f"URL: {url}")
            
            # Try to scrape
            assignment = scraper.scrape_game_umpires(game_date, home_team, away_team)
            
            if assignment:
                print(f"✅ SUCCESS!")
                print(f"   Home Plate: {assignment.home_plate_umpire}")
                print(f"   First Base: {assignment.first_base_umpire}")
                print(f"   Second Base: {assignment.second_base_umpire}")
                print(f"   Third Base: {assignment.third_base_umpire}")
                successful_tests += 1
            else:
                print(f"❌ No umpire data found")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print(f"\n📊 TEST SUMMARY:")
    print(f"Successful: {successful_tests}/{total_tests}")
    print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    
    if successful_tests > 0:
        print(f"\n✅ Scraper is working! Ready to run on all games.")
        return True
    else:
        print(f"\n❌ Scraper needs debugging before running on all games.")
        return False

if __name__ == "__main__":
    test_scraper()
