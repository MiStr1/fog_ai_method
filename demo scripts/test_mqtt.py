import paho.mqtt.client as paho
import cv2
from base64 import b64decode
import numpy as np
import json


def on_connect(client, userdata, flags, rc):
    print("CONNECT RECEIVED with code %d." % (rc))


def on_publish(client, userdata, mid):
    print("PUBLISHED")


def on_message(client, userdata, message):
    #print("message received ", str(message.payload.decode("utf-8")))
    print("message topic=", message.topic)
    print("message qos=", message.qos)
    print("message retain flag=", message.retain)
    tmp = json.loads(message.payload)
    jpg_original = b64decode(tmp['encoded_image'])
    tmp['encoded_image'] = ''
    print(json.dumps(tmp, indent=4))
    nparr = np.fromstring(jpg_original, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    cv2.imshow('output', img_np)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


client = paho.Client(transport="websockets")
client.on_connect = on_connect
client.on_publish = on_publish
client.on_message = on_message

result = client.connect("194.249.2.112", 30533)
print(client.subscribe("jobs/67965ae8-d5e3-4e82-a123-151aae8bbc5e"))
client.loop_forever()
cv2.destroyAllWindows()
