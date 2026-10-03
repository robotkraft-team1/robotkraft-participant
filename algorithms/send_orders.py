import os
import json
import time
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set("equipe01", os.environ["MQTT_PASSWORD"])
client.tls_set()

# Démarrer la boucle réseau en tâche de fond AVANT la connexion
client.loop_start()

topic = "robotkraft/equipe01/consigne"
payload = {
    "epreuve": "ADN",
    "variante": 1,
    "id": "t1-001",
    "consigne": {"ordre": ["rouge", "jaune", "bleu"]}
}

print("Connexion au broker...")
client.connect("mqtt.teleport.francsducloud.wtf", 443)

# Laisser un court instant pour stabiliser la connexion TLS
time.sleep(1)

print(f"Envoi du message sur le topic : {topic}")
info = client.publish(topic, json.dumps(payload), qos=0)

# Attendre la confirmation d'envoi de la file d'attente
info.wait_for_publish()

# Laisser le temps aux paquets de quitter la carte réseau
time.sleep(1)

print("Message envoyé ! Nettoyage...")
client.disconnect()
client.loop_stop()
