import asyncio
import sys
import os
sys.path.insert(0, '.')

async def test_endpoint():
    try:
        from app import get_dual_predictions_api
        result = await get_dual_predictions_api('2025-08-22')
        print('API endpoint test successful!')
        print('Summary:', result["summary"])
        print('Number of games:', len(result["games"]))
        if result['games']:
            game = result['games'][0]
            print('First game:', game["matchup"])
            print('  Original:', game["predictions"]["original"])
            print('  Learning:', game["predictions"]["learning"])
    except Exception as e:
        print('API endpoint test failed:', e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_endpoint())
