from pytz import utc
from pytz import timezone
from datetime import datetime


jst_tz = timezone('Asia/Tokyo')
utc_tz = timezone('UTC')

utcnow = datetime.utcnow()
# print ("utcnow1: %s"%utcnow)
utcnow = utcnow.replace(tzinfo=utc_tz)
# print ("utcnow2: %s"%utcnow)
tokyo = utcnow.astimezone(jst_tz)

print ("tokyo: %s"%tokyo)
print ("format: %s"%tokyo.strftime('%Y-%m-%d %H:%M:%S'))
