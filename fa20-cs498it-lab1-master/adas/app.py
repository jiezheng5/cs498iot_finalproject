import asyncio
import websockets
import json

from object_detection_running_cam import TensorCamera
from asgiref.sync import sync_to_async

import logging
logger = logging.getLogger('websockets')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

PORT = 9000


print("Init state...")
STATE = {'camera': TensorCamera() }
print("Initialized!")

def adas_event():
    return STATE['camera'].capture_and_tell()

async def producer_handler(websocket, path):
    while True:
        # websockets expects async. This is a hacky way
        # to avoid restructuring the detection.
        message_dict = await sync_to_async(adas_event)()

        await websocket.send(message_dict['image']) # Will send as arrayBuffer
        
        msg_json_str = json.dumps({'brake' : message_dict['brake'],
                                   'distance': message_dict['distance']})
        
        await websocket.send(msg_json_str)


print("Starting websocket server")
server = websockets.serve(producer_handler, None, PORT)
print("Websocket server started")


asyncio.get_event_loop().run_until_complete(server)
asyncio.get_event_loop().run_forever()
