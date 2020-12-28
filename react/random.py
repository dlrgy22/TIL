import time
import random
import Elasticsearch

data = []

last_time = time.time()
count = 0
while count < 10:
    if time.time() - last_time >= 10:
        last_time = time.time()
        t = time.strftime('%c', time.localtime(time.time()))
        data = random.randrange(1, 100)
        Elasticsearch.insert_data('localhost:9200', [t, data], 'random_data')
        count += 1
