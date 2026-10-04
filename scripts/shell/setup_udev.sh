#!/bin/bash
# Crée des symlinks udev stables pour les bras SO-ARM101
# Résultat : /dev/lerobot_follower et /dev/lerobot_leader
set -e

RULE_FILE="/etc/udev/rules.d/99-lerobot.rules"
VENDOR="1a86"

# Lit un attribut (KERNELS ou ATTRS{serial}) dans le bloc du device USB parent qui porte
# idVendor=$VENDOR. udev exige que KERNELS/ATTRS d'une même règle matchent le MÊME parent :
# prendre le premier KERNELS venu (interface "1-2:1.0") ou le premier serial (contrôleur USB)
# donne une règle qui ne matche jamais ou des serials identiques pour les deux bras.
usb_attr() {
    udevadm info -a -n "$1" | awk -v key="$2" -v vendor="$VENDOR" '
        /looking at parent device/ { val = "" ; isdev = 0 }
        index($0, key "==") { split($0, a, "\""); val = a[2] }
        index($0, "ATTRS{idVendor}==\"" vendor "\"") { isdev = 1 }
        isdev && val != "" { print val; exit }
        /^$/ && isdev { exit }
    '
}

echo "=== Setup udev SO-ARM101 ==="
echo ""

# Étape 1 : follower seul
echo "Débranchez les deux bras, puis branchez UNIQUEMENT le FOLLOWER."
echo "Appuyez sur Entrée quand c'est fait."
read -r

FOLLOWER_DEV=$(ls /dev/ttyACM* 2>/dev/null | head -1)
if [ -z "$FOLLOWER_DEV" ]; then
    echo "Erreur : aucun /dev/ttyACM* détecté. Vérifiez le branchement et les permissions."
    exit 1
fi
echo "  Follower détecté sur $FOLLOWER_DEV"

FOLLOWER_SERIAL=$(usb_attr "$FOLLOWER_DEV" 'ATTRS{serial}')
FOLLOWER_KERNELS=$(usb_attr "$FOLLOWER_DEV" 'KERNELS')
echo "  Serial   : ${FOLLOWER_SERIAL:-(vide)}"
echo "  USB path : $FOLLOWER_KERNELS"
echo ""

# Étape 2 : leader en plus
echo "Branchez maintenant le LEADER aussi (le follower reste branché)."
echo "Appuyez sur Entrée quand c'est fait."
read -r

LEADER_DEV=$(ls /dev/ttyACM* 2>/dev/null | grep -v "^$FOLLOWER_DEV$" | head -1)
if [ -z "$LEADER_DEV" ]; then
    echo "Erreur : impossible de distinguer le leader (aucun nouveau ttyACM apparu)."
    exit 1
fi
echo "  Leader détecté sur $LEADER_DEV"

LEADER_SERIAL=$(usb_attr "$LEADER_DEV" 'ATTRS{serial}')
LEADER_KERNELS=$(usb_attr "$LEADER_DEV" 'KERNELS')
echo "  Serial   : ${LEADER_SERIAL:-(vide)}"
echo "  USB path : $LEADER_KERNELS"
echo ""

# Choix de la méthode
if [ -n "$FOLLOWER_SERIAL" ] && [ -n "$LEADER_SERIAL" ] && [ "$FOLLOWER_SERIAL" != "$LEADER_SERIAL" ]; then
    echo "Numéros de série distincts -> règles basées sur le serial (plug-order indépendant)."
    RULE_FOLLOWER="SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"$VENDOR\", ATTRS{serial}==\"$FOLLOWER_SERIAL\", SYMLINK+=\"lerobot_follower\", MODE=\"0666\""
    RULE_LEADER="SUBSYSTEM==\"tty\",   ATTRS{idVendor}==\"$VENDOR\", ATTRS{serial}==\"$LEADER_SERIAL\",   SYMLINK+=\"lerobot_leader\",   MODE=\"0666\""
else
    echo "Serials identiques ou absents -> règles basées sur le port USB physique."
    echo "ATTENTION : branchez toujours le follower et le leader sur les mêmes ports USB."
    RULE_FOLLOWER="SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"$VENDOR\", KERNELS==\"$FOLLOWER_KERNELS\", SYMLINK+=\"lerobot_follower\", MODE=\"0666\""
    RULE_LEADER="SUBSYSTEM==\"tty\",   ATTRS{idVendor}==\"$VENDOR\", KERNELS==\"$LEADER_KERNELS\",   SYMLINK+=\"lerobot_leader\",   MODE=\"0666\""
fi

# Écriture
echo ""
echo "Écriture de $RULE_FILE (sudo requis)..."
printf '%s\n%s\n' "$RULE_FOLLOWER" "$RULE_LEADER" | sudo tee "$RULE_FILE"
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty

echo ""
echo "Vérification (débranchez/rebranchez si les symlinks n'apparaissent pas) :"
ls -la /dev/lerobot_* 2>/dev/null && echo "OK." || echo "Rebranchez les deux bras pour activer les règles."
