# Ressources techniques

> Source : https://docmost.alsacedigitale.org/share/7znmwljzor/p/ressources-techniques-9vYGPa03Ao

## Calcul

Chaque équipe a trois GPU de 24 Go pour elle (RTX 6000 Pro Blackwell), sur des serveurs distants. Les accès sont donnés sur place.

## Écosystème logiciel

L'événement repose sur LeRobot, l'écosystème open source de Hugging Face pour la robotique par apprentissage. Les bras sont entraînés **par démonstration**, pas programmés en dur.

## Un bras, une équipe, du début à la fin

Chaque équipe travaille sur le même bras pendant tout le week-end. Les servomoteurs n'ont pas de butée et la calibration réalise un zéroing propre à chaque bras : **un programme entraîné sur un bras ne se transpose pas tel quel sur un autre.**

## La contrainte à ne jamais oublier

**Au-delà de son couple maximal, le bras se met en erreur et devient flasque.** Aucune épreuve ne demande de forcer pour emboîter ou insérer, et forcer ne sert à rien.

## Brancher un bras, dans cet ordre

⚠️ Deux erreurs détruisent du matériel.

**L'alimentation d'abord, l'USB ensuite.** Si l'USB est branché avant l'alimentation, le port USB se met à alimenter la carte et peut griller. L'ordre est donc : alimentation, puis USB follower, puis USB leader. Pour débrancher, l'inverse exact : USB leader, USB follower, puis l'alimentation.

**Chaque bras a sa tension, et les connecteurs sont interchangeables.** Le follower est en 12V, le leader en 5V. Brancher le 12V sur un leader le détruit. D'où le repérage par couleur : vert, c'est le 5V, donc le leader. Blanc, c'est le 12V, donc le follower. Le fil vert sur le fil vert, le fil blanc sur le fil blanc.

## À préparer chez toi

**Docker installé, le dépôt cloné et l'image tirée** (`docker compose pull`, environ 2,7 Go), plus un compte Hugging Face et son token d'accès. Le détail est sur la page [Ce qu'il faut apporter](ce-qu-il-faut-apporter.md).

## Mise au point des caméras

La caméra ET-S231 (kits 9 à 11) se règle à la main, par une bague sur l'objectif.

Avant d'enregistrer des démonstrations, vérifie que l'image est nette à la distance où le bras saisit les objets.

Une fois le réglage fait, ne touche plus à la bague de mise au point. Si la netteté change en cours de route, les données d'entraînement deviennent incohérentes et la politique apprise se dégrade.

Pour vérifier : `make view-camera DEVICE=/dev/videoX` sous Linux, ou l'application caméra du système sous macOS et Windows (`view-camera` ne marche que sous Linux).

C'est particulièrement important pour les tâches où la caméra doit lire du texte ou distinguer de petits détails.

Références des modèles de caméra équipés sur chaque kit : voir le README du dépôt participant.

## Liens

- Dépôt GitHub participants, environnement Docker et commandes LeRobot
- Page RobotKraft
- Billetterie HelloAsso
