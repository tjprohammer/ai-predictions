#!/usr/bin/env python3
"""
Sportradar MLB API Umpire Data Research Script

Purpose: Investigate Sportradar trial access for real MLB umpire assignments
Context: Replace simulated umpire data in Phase 4 enhancement with real API data

Research Goals:
1. Test Sportradar trial API access and limits
2. Evaluate umpire data availability and format
3. Assess integration complexity with existing system
4. Compare data quality vs. simulated umpire assignments

API Endpoints to Test:
- Officials: GET /mlb/trial/v8/en/league/officials.json
- Game Boxscore: GET /mlb/trial/v8/en/games/{game_id}/boxscore.json
- Game Summary: GET /mlb/trial/v8/en/games/{game_id}/summary.json
- Play-by-Play: GET /mlb/trial/v8/en/games/{game_id}/pbp.json

Expected Umpire Data:
- HP = Home plate umpire (most critical for O/U modeling)
- 1B = 1st base umpire
- 2B = 2nd base umpire  
- 3B = 3rd base umpire
- LF = Left field umpire (postseason)
- RF = Right field umpire (postseason)

Next Steps:
1. Sign up: https://console.sportradar.com/signup
2. Get trial API key
3. Test endpoints below
4. Document findings for Phase 4 decision
"""

import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class SportradarUmpireResearcher:
    """Research Sportradar MLB API for real umpire data"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize with Sportradar trial API key
        
        Args:
            api_key: Trial API key from console.sportradar.com
        """
        self.api_key = api_key
        self.base_url = "https://api.sportradar.com/mlb/trial/v8/en"
        self.session = requests.Session()
        
        if self.api_key:
            self.session.params = {'api_key': self.api_key}
        
        self.research_results = {
            'trial_access': False,
            'officials_endpoint': {},
            'game_umpire_data': {},
            'data_quality': {},
            'integration_notes': []
        }
    
    def test_trial_access(self) -> bool:
        """Test if trial API key works"""
        try:
            url = f"{self.base_url}/league/hierarchy.json"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                logging.info("✅ Trial API access successful")
                self.research_results['trial_access'] = True
                return True
            elif response.status_code == 401:
                logging.error("❌ Invalid API key - check trial signup")
                return False
            else:
                logging.error(f"❌ API error: {response.status_code}")
                return False
                
        except Exception as e:
            logging.error(f"❌ Connection error: {e}")
            return False
    
    def research_officials_endpoint(self) -> Dict:
        """Test Officials endpoint for umpire database"""
        logging.info("🔍 Researching Officials endpoint...")
        
        try:
            url = f"{self.base_url}/league/officials.json"
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                officials = data.get('league', {}).get('officials', [])
                
                # Analyze umpire data structure
                analysis = {
                    'total_officials': len(officials),
                    'sample_official': officials[0] if officials else None,
                    'available_fields': list(officials[0].keys()) if officials else [],
                    'umpire_names': [off.get('name', 'Unknown') for off in officials[:10]]
                }
                
                logging.info(f"✅ Officials endpoint successful: {len(officials)} officials found")
                self.research_results['officials_endpoint'] = analysis
                return analysis
                
            else:
                logging.error(f"❌ Officials endpoint failed: {response.status_code}")
                return {}
                
        except Exception as e:
            logging.error(f"❌ Officials endpoint error: {e}")
            return {}
    
    def research_game_umpire_data(self, sample_game_ids: List[str] = None) -> Dict:
        """Test game endpoints for umpire assignment data"""
        logging.info("🔍 Researching game-level umpire data...")
        
        if not sample_game_ids:
            # Use some recent 2025 game IDs (would need to get from schedule first)
            sample_game_ids = []
        
        game_results = {}
        
        for game_id in sample_game_ids[:3]:  # Test a few games
            logging.info(f"Testing game {game_id}...")
            
            # Test different game endpoints
            endpoints = {
                'boxscore': f"/games/{game_id}/boxscore.json",
                'summary': f"/games/{game_id}/summary.json", 
                'pbp': f"/games/{game_id}/pbp.json"
            }
            
            game_data = {}
            for endpoint_name, endpoint_path in endpoints.items():
                try:
                    url = f"{self.base_url}{endpoint_path}"
                    response = self.session.get(url, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Look for umpire data in response
                        umpire_info = self.extract_umpire_info(data)
                        if umpire_info:
                            game_data[endpoint_name] = umpire_info
                            logging.info(f"✅ Found umpire data in {endpoint_name}")
                        else:
                            logging.info(f"⚠️  No umpire data in {endpoint_name}")
                    else:
                        logging.warning(f"❌ {endpoint_name} failed: {response.status_code}")
                        
                except Exception as e:
                    logging.error(f"❌ {endpoint_name} error: {e}")
            
            if game_data:
                game_results[game_id] = game_data
        
        self.research_results['game_umpire_data'] = game_results
        return game_results
    
    def extract_umpire_info(self, game_data: Dict) -> Dict:
        """Extract umpire information from game API response"""
        umpires = {}
        
        # Look for umpire data in various possible locations
        search_paths = [
            ['game', 'officials'],
            ['officials'],
            ['game', 'umpires'],
            ['umpires'],
            ['game', 'crew'],
            ['crew']
        ]
        
        for path in search_paths:
            current = game_data
            for key in path:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    break
            else:
                # Found potential umpire data
                if isinstance(current, list):
                    for official in current:
                        if isinstance(official, dict):
                            position = official.get('position', official.get('assignment', 'unknown'))
                            name = official.get('name', official.get('full_name', 'Unknown'))
                            umpires[position] = name
                elif isinstance(current, dict):
                    umpires.update(current)
        
        return umpires
    
    def get_recent_game_schedule(self) -> List[str]:
        """Get recent game IDs to test umpire data"""
        try:
            # Get yesterday's games
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y/%m/%d')
            url = f"{self.base_url}/games/{yesterday}/schedule.json"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                games = data.get('league', {}).get('games', [])
                game_ids = [game.get('id') for game in games if game.get('id')]
                logging.info(f"✅ Found {len(game_ids)} recent games for testing")
                return game_ids[:5]  # Return first 5 for testing
            else:
                logging.warning(f"⚠️  Could not get recent schedule: {response.status_code}")
                return []
                
        except Exception as e:
            logging.error(f"❌ Schedule error: {e}")
            return []
    
    def analyze_data_quality(self) -> Dict:
        """Analyze quality of umpire data found"""
        logging.info("📊 Analyzing umpire data quality...")
        
        analysis = {
            'has_officials_database': bool(self.research_results['officials_endpoint']),
            'has_game_umpires': bool(self.research_results['game_umpire_data']),
            'plate_umpire_coverage': 0,
            'full_crew_coverage': 0,
            'data_completeness': 'unknown',
            'vs_simulated_quality': 'unknown'
        }
        
        # Analyze game-level umpire coverage
        game_data = self.research_results['game_umpire_data']
        if game_data:
            total_games = len(game_data)
            plate_umpire_count = 0
            full_crew_count = 0
            
            for game_id, endpoints in game_data.items():
                has_plate_umpire = False
                has_full_crew = False
                
                for endpoint, umpires in endpoints.items():
                    if 'HP' in umpires or 'home_plate' in str(umpires).lower():
                        has_plate_umpire = True
                    if len(umpires) >= 4:  # Base crew
                        has_full_crew = True
                
                if has_plate_umpire:
                    plate_umpire_count += 1
                if has_full_crew:
                    full_crew_count += 1
            
            analysis['plate_umpire_coverage'] = plate_umpire_count / total_games if total_games > 0 else 0
            analysis['full_crew_coverage'] = full_crew_count / total_games if total_games > 0 else 0
        
        # Compare to our simulated approach
        if analysis['has_game_umpires'] and analysis['plate_umpire_coverage'] > 0:
            analysis['vs_simulated_quality'] = 'SUPERIOR - Real assignments vs simulated'
            analysis['data_completeness'] = 'REAL_API_DATA'
        elif analysis['has_officials_database']:
            analysis['vs_simulated_quality'] = 'POTENTIAL - Officials DB available'
            analysis['data_completeness'] = 'PARTIAL_API_DATA'
        else:
            analysis['vs_simulated_quality'] = 'EQUIVALENT - No advantage over simulation'
            analysis['data_completeness'] = 'NO_API_DATA'
        
        self.research_results['data_quality'] = analysis
        return analysis
    
    def generate_research_report(self) -> str:
        """Generate comprehensive research report"""
        report = []
        report.append("🏟️ SPORTRADAR MLB UMPIRE DATA RESEARCH REPORT")
        report.append("=" * 60)
        report.append(f"Research Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Trial Access
        report.append("📋 TRIAL ACCESS STATUS:")
        if self.research_results['trial_access']:
            report.append("   ✅ Trial API key working")
        else:
            report.append("   ❌ Trial API access failed")
            report.append("   🔧 Action: Sign up at console.sportradar.com/signup")
        report.append("")
        
        # Officials Database
        report.append("👨‍⚖️ OFFICIALS DATABASE:")
        officials = self.research_results['officials_endpoint']
        if officials:
            report.append(f"   ✅ {officials['total_officials']} officials found")
            report.append(f"   📊 Available fields: {officials['available_fields']}")
            report.append(f"   👥 Sample umpires: {officials['umpire_names'][:5]}")
        else:
            report.append("   ❌ Officials endpoint failed or no data")
        report.append("")
        
        # Game-Level Umpire Data
        report.append("⚾ GAME-LEVEL UMPIRE ASSIGNMENTS:")
        game_data = self.research_results['game_umpire_data']
        if game_data:
            report.append(f"   ✅ Tested {len(game_data)} games")
            for game_id, endpoints in game_data.items():
                report.append(f"   🎮 Game {game_id}:")
                for endpoint, umpires in endpoints.items():
                    report.append(f"      {endpoint}: {umpires}")
        else:
            report.append("   ❌ No game umpire data found")
        report.append("")
        
        # Data Quality Analysis
        report.append("📊 DATA QUALITY ASSESSMENT:")
        quality = self.research_results['data_quality']
        if quality:
            report.append(f"   📈 Plate umpire coverage: {quality['plate_umpire_coverage']:.1%}")
            report.append(f"   👥 Full crew coverage: {quality['full_crew_coverage']:.1%}")
            report.append(f"   🏆 vs Simulated data: {quality['vs_simulated_quality']}")
            report.append(f"   ✅ Data completeness: {quality['data_completeness']}")
        report.append("")
        
        # Integration Recommendations
        report.append("🚀 INTEGRATION RECOMMENDATIONS:")
        if self.research_results['trial_access'] and officials:
            if quality.get('plate_umpire_coverage', 0) > 0.8:
                report.append("   🎯 STRONGLY RECOMMENDED: High-quality real umpire data available")
                report.append("   🔧 Action: Implement SportradarUmpireCollector for Phase 4")
                report.append("   📈 Impact: Replace 100% simulated data with real assignments")
            elif quality.get('data_completeness') == 'PARTIAL_API_DATA':
                report.append("   ⚖️  CONDITIONALLY RECOMMENDED: Partial real data available")
                report.append("   🔧 Action: Hybrid approach - real where available, simulated fallback")
            else:
                report.append("   ⏸️  NOT RECOMMENDED: Limited advantage over simulation")
                report.append("   🔧 Action: Complete Phase 4 with simulated data")
        else:
            report.append("   ⏸️  BLOCKED: Trial access issues or no officials data")
            report.append("   🔧 Action: Resolve trial access, then re-evaluate")
        
        report.append("")
        report.append("📞 NEXT STEPS:")
        report.append("   1. Review this report for data quality assessment")
        report.append("   2. Make Phase 4 implementation decision")
        report.append("   3. Either implement real umpire collection or proceed with simulation")
        
        return "\n".join(report)
    
    def run_full_research(self) -> str:
        """Run complete umpire data research"""
        logging.info("🚀 Starting Sportradar umpire data research...")
        
        if not self.api_key:
            logging.error("❌ No API key provided. Sign up at: https://console.sportradar.com/signup")
            return "NO API KEY - SIGN UP REQUIRED"
        
        # Test trial access
        if not self.test_trial_access():
            return "TRIAL ACCESS FAILED"
        
        # Research officials database
        self.research_officials_endpoint()
        
        # Get recent games and test umpire data
        recent_games = self.get_recent_game_schedule()
        if recent_games:
            self.research_game_umpire_data(recent_games)
        
        # Analyze data quality
        self.analyze_data_quality()
        
        # Generate report
        report = self.generate_research_report()
        
        # Save report
        report_file = f"sportradar_umpire_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logging.info(f"📋 Research complete! Report saved: {report_file}")
        return report

def main():
    """Main research execution"""
    print("🏟️ Sportradar MLB Umpire Data Research")
    print("=" * 50)
    print()
    print("This script researches real MLB umpire data availability via Sportradar API")
    print("to determine if we should replace simulated umpire data in Phase 4.")
    print()
    print("REQUIRED: Sportradar trial API key")
    print("Sign up: https://console.sportradar.com/signup")
    print()
    
    api_key = input("Enter your Sportradar trial API key (or press Enter to skip): ").strip()
    
    if not api_key:
        print("⏸️  Skipping research - no API key provided")
        print("   1. Sign up at: https://console.sportradar.com/signup")
        print("   2. Get trial API key")
        print("   3. Run this script again with the key")
        return
    
    # Run research
    researcher = SportradarUmpireResearcher(api_key)
    report = researcher.run_full_research()
    
    print("\n" + "="*60)
    print("📋 RESEARCH COMPLETE!")
    print("="*60)
    print(report)

if __name__ == "__main__":
    main()
