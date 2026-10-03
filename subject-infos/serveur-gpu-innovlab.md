# Se connecter au serveur GPU InnovLab

> Source : https://docmost.alsacedigitale.org/share/7znmwljzor/p/se-connecter-au-serveur-gpu-innov-lab-FqGblXvd2H

L'InnovLab est le laboratoire de **TPS**, Télécom Physique Strasbourg. Il met un second parc GPU à disposition du challenge, en complément de celui d'Agora.

> **Deux accès, deux procédures.** Agora passe par Teleport et par un compte personnel, et fonctionne de n'importe où. L'InnovLab est un JupyterHub joignable **depuis le réseau de Fabéon**, avec des identifiants fournis. Vérifié depuis Fabéon le 1er octobre. Depuis chez toi ou en partage de connexion mobile, ne compte pas dessus : l'accès a été ouvert pour l'adresse du lieu. Le code du réseau wifi est donné pendant la présentation, vendredi soir.

## 1. Ouvre le serveur

<https://130.79.207.171:5557>

Ton navigateur va probablement afficher un **avertissement de sécurité**. C'est normal : le serveur est joint par son adresse IP, son certificat ne correspond donc à aucun nom de domaine. Passe outre et continue.

## 2. Connecte-toi

Saisis les identifiants qui t'ont été fournis.

## 3. Crée ton instance

Une seule configuration est proposée : GPU A100 MIG 20GB, 2g.20gb (4 cores, 16 Go de RAM, 4 h maximum). Sélectionne-la, puis clique sur Start.

Le démarrage prend un moment, c'est normal. Une fois l'instance prête, la redirection se fait automatiquement vers JupyterLab.

## 4. Libère l'instance quand tu as fini

C'est important, les ressources sont partagées.

**File**, puis **Hub Control Panel**, puis **Stop My Server**.

Tant que ton instance tourne, elle occupe du GPU même si tu ne t'en sers pas. Prends le réflexe de l'arrêter avant une pause longue ou en fin de journée.

## Différences avec le serveur Agora

|            | Agora                              | InnovLab                                       |
| ---------- | ---------------------------------- | ---------------------------------------------- |
| Accès      | Teleport, de n'importe où          | JupyterHub, depuis le réseau de Fabéon         |
| Authentification | Compte Teleport, avec second facteur | Identifiants fournis                          |
| GPU        | RTX 6000 Pro Blackwell             | A100, 20 Go                                    |
| Mise en route | Lancement du JupyterHub de ton équipe | Création d'une instance, puis arrêt manuel   |

## Si ça coince

- **La page ne charge pas** : vérifie que tu es bien sur le réseau de Fabéon, et non en partage de connexion mobile. L'accès est ouvert pour l'adresse du lieu.
- **L'avertissement de sécurité bloque** : cherche « Paramètres avancés » puis « Continuer vers le site ». Le certificat est auto-signé, ce n'est pas une attaque.
- **L'instance ne démarre pas** : quelqu'un occupe peut-être les ressources. Signale-le plutôt que de relancer en boucle.

Dans tous les cas, le salon `support-technique` du Discord, ou quelqu'un de l'équipe coaching dans la salle.
Côté organisation, Christophe Heinkele porte l'accès InnovLab.
