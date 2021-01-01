import time
import random
import Elasticsearch

from datetime import datetime

def utc_time():
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

data = []

last_time = time.time()
count = 0
while count < 5:
    if time.time() - last_time >= 10:
        last_time = time.time()
        t = utc_time()
        data = random.randrange(1, 100)
        Elasticsearch.insert_data('localhost:9200', [t, data], 'fire')
        count += 1
