# Se connecter au serveur GPU Agora

> Source : https://docmost.alsacedigitale.org/share/7znmwljzor/p/se-connecter-au-serveur-gpu-agora-WEI6veP2Ft

Le serveur GPU d'**Agora Calycé** est derrière **Teleport**. On n'y accède pas par un simple mot de passe : il faut créer son compte depuis un lien d'invitation, puis passer par une authentification à deux facteurs. Comptez cinq minutes, une fois pour toutes.

> Cette page ne concerne que l'accès **Agora**. L'accès aux machines de **TPS et d'Innovlab** fait l'objet d'une documentation distincte.

## Ce qu'il te faut avant de commencer

- Le **lien d'invitation** reçu par mail. **Ouvre-le tout de suite** : il expire vite, en une heure pour une première invitation, en huit heures s'il s'agit de réinitialiser un accès existant. Passé ce délai il faut en redemander un, c'est immédiat mais ça suppose que quelqu'un soit disponible pour le faire.
- De quoi faire une **authentification à deux facteurs** : Touch ID, Face ID, Windows Hello, une clé physique, ou une application comme Google Authenticator ou Authy.

## 1. Ouvre le lien, et lance la procédure

Le lien t'amène sur une page d'accueil Teleport. Clique sur **Continue**.

Selon la façon dont ton compte a été créé, le titre affiché est soit **Welcome**, soit **Reset Authentication**. Le mot « Reset » est trompeur quand c'est un premier accès, mais tu es au bon endroit dans les deux cas.

## 2. Choisis ton mot de passe (étape 1 sur 2)

La page **Set a Password** affiche ton nom d'utilisateur, de la forme `prenom.nom`. Il est déjà rempli, ne le modifie pas : c'est l'identifiant de ton compte.

Saisis ton mot de passe deux fois, puis **Next**.

> Le bouton **Go Passwordless** permet de se passer de mot de passe et de n'utiliser que la biométrie ou une clé. Ça fonctionne, mais en cas de souci le dépannage est plus compliqué. Pour un week-end, prends un mot de passe.

## 3. Ajoute un second facteur (étape 2 sur 2)

La page **Set up Multi-Factor Authentication** propose deux familles :

- **Passkey ou clé de sécurité** : Touch ID, Face ID, Windows Hello, ou une clé physique. C'est le choix par défaut, et le plus rapide si ta machine le gère.
- **Application d'authentification** : un code à six chiffres renouvelé toutes les trente secondes, type Google Authenticator, Authy, 1Password.

Clique sur **Create an MFA Method** et laisse-toi guider par ton système.

> ⚠️ Si tu choisis Touch ID ou Windows Hello, ton second facteur est **lié à cette machine**. Depuis un autre ordinateur, tu ne pourras pas te connecter. Si tu comptes changer de poste pendant le week-end, prends plutôt une application d'authentification, qui te suit sur ton téléphone.

## 4. C'est fait

Un écran **Reset Successful** confirme la création. Clique sur **Go to Cluster**.

## 5. Trouve ton environnement de travail

Tu arrives sur la liste **Resources**, où tu vois les JupyterHub des équipes, `equipe01` à `equipe10`, et les serveurs en accès SSH.

Clique sur **Launch** à droite de la ligne de ton équipe. Un nouvel onglet s'ouvre sur une adresse du type `https://equipe09.teleport.francsducloud.wtf`.

> **Tu ne sais pas encore quel est ton numéro d'équipe ?** C'est normal : les équipes se constituent le vendredi soir, à l'accueil. Ton numéro te sera communiqué à ce moment-là. Tu peux créer ton compte avant, les étapes 1 à 4 ne dépendent pas de l'équipe.

## 6. Tu es dans JupyterLab

L'écran **Launcher** te propose de créer un notebook Python 3, une console, un terminal ou un fichier texte.

Dans le panneau de gauche, trois dossiers :

- `datasets` : les jeux de données
- `equipes` : l'espace partagé entre équipes
- `work` : ton espace de travail

Un terminal est disponible dans la section **Other** du Launcher : c'est là que tu retrouves une ligne de commande sur le serveur, avec le GPU.

## Si ça coince

- **Le lien ne fonctionne plus** : il a expiré, une heure pour une première invitation. Demande-en un nouveau, c'est immédiat.
- **Tu ne vois pas le JupyterHub de ton équipe** dans Resources : soit les équipes ne sont pas encore constituées, soit ton compte n'est pas rattaché à la bonne. Signale-le.
- **Tu as changé d'ordinateur et tu ne peux plus te connecter** : ton second facteur est lié à l'autre machine. Il faut réinitialiser ton accès.

Dans tous les cas, le salon `support-technique` du Discord, ou quelqu'un de l'équipe coaching dans la salle.
