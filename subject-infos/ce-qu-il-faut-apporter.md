# Ce qu'il faut apporter

> Source : https://docmost.alsacedigitale.org/share/7znmwljzor/p/ce-qu-il-faut-apporter-1Mimdlr2G5

Découvrir cette page le vendredi soir fait perdre deux heures. Elle se lit en deux minutes.

## À apporter impérativement

- **Ton ordinateur portable** et son chargeur.
- **Un hub USB d'au moins 3 ports.** Le bras et sa caméra les occupent tous les trois.
- **Une multiprise.**

## Fourni sur place

- Les bras robotisés SO-ARM101 et leurs caméras.
- L'accès aux serveurs GPU distants.
- Le matériel propre à chaque épreuve.

## Préparation logicielle, à faire chez toi

Trois choses avant de venir. Les découvrir sur place coûte une soirée.

1. **Installe Docker.** L'installation demande les droits administrateur, et parfois un redémarrage.
2. **Clone le dépôt et tire l'image.** Environ 2,7 Go sur disque. Des dizaines de téléchargements simultanés sur la connexion du lieu, ce n'est pas une bonne idée.
3. **Crée un compte Hugging Face et génère un token d'accès.** Il est demandé dans le fichier `.env` pour récupérer et publier datasets et modèles.

```bash
git clone https://github.com/AlsaceDigitale/robotkraft-participant.git
cd robotkraft-participant
docker compose pull
```

## Quel système d'exploitation

**L'environnement est testé sous Linux.** Sous macOS et sous Windows, l'accès USB aux bras passe par d'autres chemins. On t'aidera sur place à le faire fonctionner, mais prévois du temps, et viens avec Docker déjà installé.

## Un point à connaître dès l'arrivée

**Les caméras ne sont pas les mêmes d'un poste à l'autre.** C'est le seul élément réellement non portable d'un bras à l'autre.
