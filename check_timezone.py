from datetime import datetime

# Check what the JavaScript logic would produce
now_local = datetime.now()
now_utc = datetime.utcnow()

print('🕐 DATE CALCULATION ANALYSIS:')
print('=' * 40)
print(f'Local time: {now_local.year}-{now_local.month:02d}-{now_local.day:02d} {now_local.hour:02d}:{now_local.minute:02d}')
print(f'UTC time: {now_utc.year}-{now_utc.month:02d}-{now_utc.day:02d} {now_utc.hour:02d}:{now_utc.minute:02d}')

# Simulate JavaScript new Date().toISOString().split('T')[0]
js_date = f'{now_utc.year}-{now_utc.month:02d}-{now_utc.day:02d}'
py_date = f'{now_local.year}-{now_local.month:02d}-{now_local.day:02d}'

print(f'JavaScript logic (UTC): {js_date}')
print(f'Python logic (Local): {py_date}')

if js_date != py_date:
    print('🚨 TIMEZONE MISMATCH FOUND!')
    print('   UI is using UTC date, API is using local date')
else:
    print('✅ Dates match')
