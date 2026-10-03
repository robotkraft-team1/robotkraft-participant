import os
import json
import typing as T
import paho.mqtt.client as mqtt            # paho-mqtt >= 2
from dotenv import load_dotenv
import adn

load_dotenv()

derniere_id = None

def on_message(client: mqtt.Client, userdata: T.Any, msg: mqtt.MQTTMessage):
    print("message")
    global derniere_id
    data = json.loads(msg.payload)
    if data.get("id") == derniere_id:
        return  # already handled
    derniere_id = data["id"]
    consigne = data["consigne"]
    epreuve = data["epreuve"]

    print(epreuve)
    
    if epreuve == "ADN":
        adn.main_task(consigne["ordre"])

c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
c.username_pw_set("equipe01", os.environ["MQTT_PASSWORD"])
c.tls_set()
c.on_message = on_message
c.connect("mqtt.teleport.francsducloud.wtf", 443)
c.subscribe("robotkraft/equipe01/consigne")
c.loop_forever()