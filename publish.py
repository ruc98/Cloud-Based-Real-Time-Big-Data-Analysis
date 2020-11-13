from google.cloud import storage 
from google.cloud import pubsub_v1
import time

publisher = pubsub_v1.PublisherClient()
## Topic Name
topic_path = publisher.topic_path('big-data-lab-266810', 'to-kafka')
client = storage.Client()
bucket = client.get_bucket('wcs_word')
blob = bucket.get_blob('yelp_test.csv')
x = blob.download_as_string()
x = x.decode("utf-8-sig")
x = x.split("\n")
futures = dict()

def get_callback(f, data):
    def callback(f):
        try:
            print(f.results())
        except:
            print("Please handle {} for {}.".format(f.exception(), data))
    return callback

### Send Data after Parsing

for i in range(len(x)):
    data = x[i]
    ### removing empty datapoints
    if(len(data) < 10):
        continue
    futures.update({data: None})
    future = publisher.publish(topic_path, b'hi', val = data)
    futures[data] = future
    print(x[i])
    future.add_done_callback(get_callback(future, data))
    time.sleep(0.5)

print("Published Message with Error Handler")
