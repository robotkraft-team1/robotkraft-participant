#!/usr/bin/env python3
"""
Script de pré-annotation automatique et de vérification pour RobotKraft.

Fonctionnalités :
1. Prédit les boîtes englobantes ('tube', 'rack') sur les images de `dataset_raw` via YOLOv8.
2. Identifie la couleur du liquide pour chaque tube (JAUNE, CYAN, MAGENTA, TRANSPARENT).
3. Sauvegarde dans un nouveau dossier :
   - `labels/` : Annotations au format standard YOLO (.txt) prêtes pour l'entraînement.
   - `images/` : Liens symboliques vers les images sources (format dataset standard YOLO).
   - `visualized/` : Images avec boîtes colorées, étiquettes, couleur de liquide et scores.
   - `crops/` : Vignettes découpées de chaque objet détecté classées par type/couleur.
   - `summary.json` : Statistiques détaillées de détection + liste des images sans détection.
   - `report.html` : Galerie web interactive pour vérifier visuellement toutes les images (avec zoom & flèches).
4. Mode interactif OpenCV (--view) pour inspecter et valider image par image avec touches clavier.
5. Option --approve pour exporter directement les labels vérifiés vers `annotations/`.

Utilisation :
  .venv/bin/python oak_tools/pre_annotate.py
  .venv/bin/python oak_tools/pre_annotate.py --conf 0.25 --view
  .venv/bin/python oak_tools/pre_annotate.py --output-dir dataset_preannotated
  .venv/bin/python oak_tools/pre_annotate.py --approve
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ==============================================================================
# 1. Analyse Colorimétrique du Liquide (Identique à detect_tubes_3d.py)
# ==============================================================================
def classify_liquid_color(bgr_crop):
    """
    Analyse la couleur du tube / liquide au centre du tube :
    - TRANSPARENT : tube à essai vide (faible saturation et chromaticité)
    - JAUNE, CYAN, MAGENTA : liquide coloré
    Retourne (nom_couleur, couleur_bgr_pour_affichage)
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return "INCONNU", (128, 128, 128)

    # Chromaticité (écart max - min des canaux BGR)
    chroma = np.max(bgr_crop, axis=2).astype(float) - np.min(bgr_crop, axis=2).astype(float)
    med_chroma = float(np.median(chroma))

    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    hue = hsv[:, :, 0]

    # Masque des pixels saturés et suffisamment lumineux
    mask_sat = (sat > 40) & (val > 40)
    sat_ratio = float(np.mean(mask_sat))
    med_sat = float(np.median(sat))

    # Tube à essai vide / transparent : peu de saturation et chromaticité quasi nulle
    if med_chroma < 30 or sat_ratio < 0.20 or med_sat < 35:
        return "TRANSPARENT", (200, 200, 200)

    hues = hue[mask_sat]
    if len(hues) == 0:
        return "TRANSPARENT", (200, 200, 200)

    med_hue = float(np.median(hues))

    # Plages de teinte OpenCV (0 - 180)
    if 15 <= med_hue <= 55:
        return "JAUNE", (0, 230, 255)       # BGR Jaune vif
    elif 75 <= med_hue <= 135:
        return "CYAN", (255, 220, 0)        # BGR Cyan
    else:
        return "MAGENTA", (50, 50, 255)     # BGR Magenta / Rouge


# ==============================================================================
# 2. Utilitaires de Recherche Automatique des Chemins
# ==============================================================================
def find_first_existing(candidates):
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


def resolve_default_paths():
    base_proj = "/home/plr/PycharmProjects"

    raw_candidates = [
        "dataset_raw/all",
        "dataset_raw",
        "oak_tools/dataset_raw/all",
        "oak_tools/dataset_raw",
        "../oak_tools/dataset_raw/all",
        "../oak_tools/dataset_raw",
        os.path.join(base_proj, "oak_tools", "dataset_raw", "all"),
        os.path.join(base_proj, "oak_tools", "dataset_raw"),
    ]
    raw_dir = find_first_existing(raw_candidates)

    weights_candidates = [
        "weights/best.pt",
        "oak_tools/weights/best.pt",
        "../oak_tools/weights/best.pt",
        os.path.join(base_proj, "oak_tools", "weights", "best.pt"),
        os.path.join(base_proj, "weights", "best.pt"),
        "yolov8n.pt",
    ]
    weights_path = find_first_existing(weights_candidates)

    if os.path.isdir(os.path.join(base_proj, "oak_tools")):
        default_out = os.path.join(base_proj, "oak_tools", "dataset_preannotated")
    else:
        default_out = os.path.abspath("dataset_preannotated")

    return raw_dir, weights_path, default_out


# ==============================================================================
# 3. Dessin des Annotations
# ==============================================================================
def draw_detection_box(img, x1, y1, x2, y2, label_text, box_color, text_color=(0, 0, 0)):
    """Dessine une boîte englobante nette avec badge d'en-tête lisible."""
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2, lineType=cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

    badge_y1 = max(0, y1 - th - baseline - 6)
    badge_y2 = y1
    badge_x1 = x1
    badge_x2 = min(img.shape[1], x1 + tw + 8)

    cv2.rectangle(img, (badge_x1, badge_y1), (badge_x2, badge_y2), box_color, -1)
    cv2.putText(
        img,
        label_text,
        (badge_x1 + 4, badge_y2 - baseline - 2),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


# ==============================================================================
# 4. Générateur du Rapport HTML de Vérification
# ==============================================================================
def generate_html_report(output_dir, metadata_list, stats):
    """Génère un tableau de bord HTML moderne pour inspecter et vérifier visuellement toutes les pré-annotations."""
    html_path = os.path.join(output_dir, "report.html")

    cards_html = []
    for idx, item in enumerate(metadata_list):
        fname = item["filename"]
        cam = item["camera"]
        det_count = item["num_detections"]
        tubes_cnt = item["num_tubes"]
        racks_cnt = item["num_racks"]
        colors = item.get("tube_colors", {})

        color_badges = "".join(
            [f'<span class="badge badge-{c.lower()}">{c}: {n}</span>' for c, n in colors.items()]
        )

        filter_classes = f"cam-{cam.lower()}"
        if det_count == 0:
            filter_classes += " is-empty"
        else:
            filter_classes += " has-detections"
            if tubes_cnt > 0:
                filter_classes += " has-tubes"
            if racks_cnt > 0:
                filter_classes += " has-racks"

        vis_rel = f"visualized/{fname}"
        lbl_rel = f"labels/{os.path.splitext(fname)[0]}.txt"

        card = f"""
        <div class="card {filter_classes}" data-index="{idx}" data-file="{fname}">
            <div class="card-img-wrapper" onclick="openModalByIndex({idx})">
                <img src="{vis_rel}" alt="{fname}" loading="lazy">
                <span class="cam-tag cam-tag-{cam.lower()}">{cam}</span>
                <span class="count-tag">{det_count} dét.</span>
            </div>
            <div class="card-body">
                <div class="card-title" title="{fname}">{fname}</div>
                <div class="card-stats">
                    <span class="stat-item">Tubes: <strong>{tubes_cnt}</strong></span>
                    <span class="stat-item">Racks: <strong>{racks_cnt}</strong></span>
                </div>
                <div class="color-badges">{color_badges}</div>
                <div class="card-actions">
                    <button onclick="openModalByIndex({idx})" class="btn-sm btn-view">Agrandir</button>
                    <a href="{lbl_rel}" target="_blank" class="btn-sm">Label .txt</a>
                </div>
            </div>
        </div>
        """
        cards_html.append(card)

    all_cards_str = "\n".join(cards_html)
    items_json = json.dumps(metadata_list, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RobotKraft - Vérification des Pré-Annotations</title>
    <style>
        :root {{
            --bg-color: #12141a;
            --surface-color: #1a1e27;
            --surface-hover: #222733;
            --primary: #3b82f6;
            --primary-hover: #2563eb;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --border: #2d3748;
            --accent-tube: #10b981;
            --accent-rack: #a855f7;
            --accent-yellow: #eab308;
            --accent-cyan: #06b6d4;
            --accent-magenta: #ec4899;
            --accent-trans: #9ca3af;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            padding: 24px;
            line-height: 1.5;
        }}
        header {{
            max-width: 1400px;
            margin: 0 auto 24px;
        }}
        .header-title {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
            margin-bottom: 20px;
        }}
        h1 {{ font-size: 1.8rem; font-weight: 700; color: #fff; }}
        .header-meta {{ color: var(--text-muted); font-size: 0.9rem; }}
        
        .kpi-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--surface-color);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}
        .kpi-val {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; }}
        .kpi-label {{ font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
        .kpi-tubes {{ color: var(--accent-tube); }}
        .kpi-racks {{ color: var(--accent-rack); }}
        .kpi-empty {{ color: #ef4444; }}

        .toolbar {{
            max-width: 1400px;
            margin: 0 auto 24px;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            background: var(--surface-color);
            padding: 14px 20px;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        .filter-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .btn-filter {{
            background: var(--bg-color);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 0.88rem;
            cursor: pointer;
            transition: all 0.15s;
        }}
        .btn-filter:hover {{ background: var(--surface-hover); border-color: var(--primary); }}
        .btn-filter.active {{ background: var(--primary); border-color: var(--primary); color: #fff; }}
        .btn-filter.btn-warning.active {{ background: #ef4444; border-color: #ef4444; }}

        .instructions-box {{
            max-width: 1400px;
            margin: 0 auto 24px;
            background: #1e293b;
            border-left: 4px solid var(--primary);
            padding: 16px 20px;
            border-radius: 4px;
            font-size: 0.9rem;
        }}
        .instructions-box code {{
            background: #0f172a;
            padding: 2px 6px;
            border-radius: 4px;
            color: #38bdf8;
            font-family: monospace;
        }}

        .grid {{
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: var(--surface-color);
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            transition: transform 0.2s, border-color 0.2s;
        }}
        .card:hover {{
            transform: translateY(-3px);
            border-color: #4b5563;
        }}
        .card.is-empty {{
            border-color: rgba(239, 68, 68, 0.5);
            background: rgba(239, 68, 68, 0.04);
        }}
        .card-img-wrapper {{
            position: relative;
            cursor: pointer;
            background: #000;
            padding-top: 56.25%; /* 16:9 */
            overflow: hidden;
        }}
        .card-img-wrapper img {{
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            object-fit: cover;
            transition: transform 0.25s;
        }}
        .card-img-wrapper:hover img {{ transform: scale(1.04); }}
        .cam-tag {{
            position: absolute;
            top: 8px;
            left: 8px;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .cam-tag-oak {{ background: #2563eb; color: #fff; }}
        .cam-tag-wrist {{ background: #7c3aed; color: #fff; }}
        .count-tag {{
            position: absolute;
            top: 8px;
            right: 8px;
            background: rgba(0,0,0,0.75);
            backdrop-filter: blur(4px);
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        .card-body {{ padding: 14px; display: flex; flex-direction: column; flex-grow: 1; }}
        .card-title {{
            font-size: 0.85rem;
            font-family: monospace;
            color: #d1d5db;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-bottom: 8px;
        }}
        .card-stats {{
            display: flex;
            gap: 12px;
            font-size: 0.85rem;
            margin-bottom: 8px;
        }}
        .stat-item strong {{ color: #fff; }}
        .color-badges {{
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            margin-bottom: 12px;
            min-height: 24px;
        }}
        .badge {{
            font-size: 0.72rem;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
        }}
        .badge-jaune {{ background: rgba(234, 179, 8, 0.2); color: #fde047; border: 1px solid rgba(234, 179, 8, 0.4); }}
        .badge-cyan {{ background: rgba(6, 182, 212, 0.2); color: #67e8f9; border: 1px solid rgba(6, 182, 212, 0.4); }}
        .badge-magenta {{ background: rgba(236, 72, 153, 0.2); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.4); }}
        .badge-transparent {{ background: rgba(156, 163, 175, 0.2); color: #d1d5db; border: 1px solid rgba(156, 163, 175, 0.4); }}

        .card-actions {{
            display: flex;
            gap: 8px;
            margin-top: auto;
        }}
        .btn-sm {{
            flex: 1;
            text-align: center;
            background: #272f3f;
            color: #93c5fd;
            text-decoration: none;
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 500;
            border: none;
            cursor: pointer;
            transition: background 0.15s;
        }}
        .btn-sm:hover {{ background: #374151; color: #fff; }}
        .btn-view {{ background: #1e3a8a; color: #bfdbfe; }}
        .btn-view:hover {{ background: #2563eb; color: #fff; }}

        /* Modal Lightbox */
        .modal {{
            display: none;
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            background: rgba(0,0,0,0.94);
            z-index: 9999;
            align-items: center;
            justify-content: center;
            flex-direction: column;
        }}
        .modal.active {{ display: flex; }}
        .modal-body {{
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            max-width: 96vw;
            max-height: 88vh;
        }}
        .modal img {{
            max-width: 90vw;
            max-height: 82vh;
            border-radius: 6px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.8);
            object-fit: contain;
        }}
        .modal-nav {{
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(30, 41, 59, 0.75);
            border: 1px solid var(--border);
            color: #fff;
            font-size: 2rem;
            padding: 12px 18px;
            border-radius: 50%;
            cursor: pointer;
            transition: background 0.2s;
            user-select: none;
        }}
        .modal-nav:hover {{ background: var(--primary); }}
        .modal-prev {{ left: 16px; }}
        .modal-next {{ right: 16px; }}

        .modal-title {{
            color: #fff;
            margin-top: 14px;
            font-family: monospace;
            font-size: 1.05rem;
            display: flex;
            gap: 16px;
            align-items: center;
        }}
        .modal-close {{
            position: absolute;
            top: 16px;
            right: 24px;
            font-size: 2.2rem;
            color: #fff;
            cursor: pointer;
            opacity: 0.8;
        }}
        .modal-close:hover {{ opacity: 1; }}
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <div>
                <h1>RobotKraft - Validation des Pré-Annotations</h1>
                <div class="header-meta">Date : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | Modèle : <code>{os.path.basename(stats['weights'])}</code> | Confiance &ge; {stats['conf']}</div>
            </div>
        </div>

        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-val">{stats['total_images']}</div>
                <div class="kpi-label">Images Traitées</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val">{stats['total_detections']}</div>
                <div class="kpi-label">Objets Détectés</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val kpi-tubes">{stats['total_tubes']}</div>
                <div class="kpi-label">Tubes Détectés</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val kpi-racks">{stats['total_racks']}</div>
                <div class="kpi-label">Racks Détectés</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val kpi-empty">{stats['empty_images']}</div>
                <div class="kpi-label">Sans Détection</div>
            </div>
        </div>

        <div class="instructions-box">
            <strong>Prochaine étape après vérification visuelle :</strong><br>
            Si les annotations vous conviennent, validez-les directement avec :<br>
            <code>python oak_tools/pre_annotate.py --approve</code><br>
            ou manuellement :<br>
            <code>cp -r {os.path.abspath(output_dir)}/labels {os.path.abspath(os.path.dirname(output_dir))}/annotations/labels_validees_{datetime.now().strftime('%Y%m%d_%H%M%S')}</code><br>
            puis relancez l'entraînement YOLO : <code>python oak_tools/train_yolo.py</code>
        </div>

        <div class="instructions-box" style="margin-top: -12px; border-left-color: #10b981;">
            <strong>Import direct dans MakeSense.ai (Fichier unique) :</strong><br>
            Fichier COCO généré : <code>{os.path.abspath(os.path.join(output_dir, 'makesense_annotations.json'))}</code> (<a href="makesense_annotations.json" download style="color: #6ee7b7; font-weight: 600;">Télécharger</a>)<br>
            1. Ouvrez <a href="https://www.makesense.ai/" target="_blank" style="color: #6ee7b7; font-weight: 600;">makesense.ai</a> et glissez vos photos.<br>
            2. Cliquez sur <em>Object Detection</em> puis <em>Rect</em>.<br>
            3. Dans le menu haut gauche : <strong>Actions &rarr; Import annotations</strong>.<br>
            4. Choisissez l'option : <strong>Single file in COCO JSON format</strong>.<br>
            5. Glissez le fichier <code>makesense_annotations.json</code> : toutes les boîtes et étiquettes sont chargées instantanément !
        </div>
    </header>

    <div class="toolbar">
        <div class="filter-group">
            <button class="btn-filter active" onclick="filterCards('all')">Toutes ({stats['total_images']})</button>
            <button class="btn-filter" onclick="filterCards('has-detections')">Avec détections ({stats['total_images'] - stats['empty_images']})</button>
            <button class="btn-filter btn-warning" onclick="filterCards('is-empty')">⚠️ Sans détection ({stats['empty_images']})</button>
            <button class="btn-filter" onclick="filterCards('has-tubes')">Avec tubes</button>
            <button class="btn-filter" onclick="filterCards('has-racks')">Avec racks</button>
            <button class="btn-filter" onclick="filterCards('cam-oak')">OAK ({stats['oak_images']})</button>
            <button class="btn-filter" onclick="filterCards('cam-wrist')">Poignet ({stats['wrist_images']})</button>
        </div>
    </div>

    <div class="grid" id="imageGrid">
        {all_cards_str}
    </div>

    <!-- Modal Lightbox -->
    <div class="modal" id="modal" onclick="closeModal()">
        <span class="modal-close">&times;</span>
        <button class="modal-nav modal-prev" onclick="prevModal(event)">&lsaquo;</button>
        <button class="modal-nav modal-next" onclick="nextModal(event)">&rsaquo;</button>
        <div class="modal-body" onclick="event.stopPropagation()">
            <img id="modalImg" src="" alt="Agrandissement">
        </div>
        <div class="modal-title" id="modalTitle"></div>
    </div>

    <script>
        const items = {items_json};
        let currentModalIdx = 0;

        function filterCards(filter) {{
            document.querySelectorAll('.btn-filter').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');

            const cards = document.querySelectorAll('.card');
            cards.forEach(card => {{
                if (filter === 'all') {{
                    card.style.display = 'flex';
                }} else if (card.classList.contains(filter)) {{
                    card.style.display = 'flex';
                }} else {{
                    card.style.display = 'none';
                }}
            }});
        }}

        function openModalByIndex(idx) {{
            if (idx < 0 || idx >= items.length) return;
            currentModalIdx = idx;
            const item = items[idx];
            const modal = document.getElementById('modal');
            const modalImg = document.getElementById('modalImg');
            const modalTitle = document.getElementById('modalTitle');

            modalImg.src = 'visualized/' + item.filename;
            modalTitle.innerHTML = `<span>[${{idx + 1}}/${{items.length}}] ${{item.filename}}</span> &bull; <span>${{item.num_detections}} détections (${{item.num_tubes}} tubes, ${{item.num_racks}} racks)</span>`;
            modal.classList.add('active');
        }}

        function closeModal() {{
            document.getElementById('modal').classList.remove('active');
        }}

        function nextModal(e) {{
            if (e) e.stopPropagation();
            if (currentModalIdx < items.length - 1) {{
                openModalByIndex(currentModalIdx + 1);
            }}
        }}

        function prevModal(e) {{
            if (e) e.stopPropagation();
            if (currentModalIdx > 0) {{
                openModalByIndex(currentModalIdx - 1);
            }}
        }}

        document.addEventListener('keydown', (e) => {{
            const modal = document.getElementById('modal');
            if (!modal.classList.contains('active')) return;
            if (e.key === 'Escape') closeModal();
            else if (e.key === 'ArrowRight' || e.key === ' ') nextModal();
            else if (e.key === 'ArrowLeft') prevModal();
        }});
    </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return html_path


# ==============================================================================
# 5. Visualiseur Interactif OpenCV
# ==============================================================================
def run_interactive_viewer(output_dir, metadata_list):
    """Permet de faire défiler toutes les images annotées au clavier et d'éliminer les faux positifs."""
    vis_dir = os.path.join(output_dir, "visualized")
    lbl_dir = os.path.join(output_dir, "labels")

    if not metadata_list:
        print("[VIEWER] Aucune image à afficher.")
        return

    idx = 0
    window_name = "RobotKraft - Verification des Pre-Annotations (Touches: n/ESPACE=Suiv, p=Prec, d=Suppr label, q=Quitter)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n" + "=" * 65)
    print("  VISUALISEUR INTERACTIF OUVERT")
    print("  Commandes :")
    print("    [ESPACE] ou [n] ou [->] : Image suivante")
    print("    [p] ou [<-]              : Image précédente")
    print("    [d]                      : Supprimer le fichier d'annotation .txt")
    print("    [q] ou [ECHAP]           : Quitter le visualiseur")
    print("=" * 65 + "\n")

    while 0 <= idx < len(metadata_list):
        item = metadata_list[idx]
        fname = item["filename"]
        vis_path = os.path.join(vis_dir, fname)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")

        img = cv2.imread(vis_path)
        if img is None:
            idx += 1
            continue

        h, w = img.shape[:2]
        display_img = img.copy()

        # Barre d'état en bas
        bar_h = 44
        overlay = display_img.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, display_img, 0.15, 0, display_img)

        exists = os.path.exists(lbl_path)
        status_txt = f"[{idx + 1}/{len(metadata_list)}] {fname} | Detections: {item['num_detections']} (Tubes: {item['num_tubes']}, Racks: {item['num_racks']})"
        if not exists:
            status_txt += " | [ANNOTATION SUPPRIMEE]"
            text_color = (0, 0, 255)
        else:
            text_color = (255, 255, 255)

        cv2.putText(display_img, status_txt, (16, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)

        # Redimensionnement adapté à l'écran si nécessaire (max 1600x900)
        max_w, max_h = 1600, 900
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            view_w = int(w * scale)
            view_h = int(h * scale)
            display_img = cv2.resize(display_img, (view_w, view_h), interpolation=cv2.INTER_AREA)

        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(0) & 0xFF

        if key in [ord("q"), 27]:
            break
        elif key in [ord(" "), ord("n"), 83]:
            idx = min(len(metadata_list) - 1, idx + 1)
        elif key in [ord("p"), 81]:
            idx = max(0, idx - 1)
        elif key == ord("d"):
            if os.path.exists(lbl_path):
                os.remove(lbl_path)
                print(f"[SUPPRIME] Annotation retirée pour : {fname}")
            else:
                print(f"[INFO] Aucun label existant pour : {fname}")

    cv2.destroyAllWindows()


# ==============================================================================
# 6. Approbation et Export Direct
# ==============================================================================
def approve_and_export(output_dir):
    """Exporte les labels pré-annotés vers annotations/ pour ré-entraînement."""
    out_dir = os.path.abspath(output_dir)
    lbl_dir = os.path.join(out_dir, "labels")

    if not os.path.exists(lbl_dir):
        print(f"[ERREUR] Le dossier de labels '{lbl_dir}' n'existe pas. Lancez d'abord la pré-annotation.")
        sys.exit(1)

    labels = glob.glob(os.path.join(lbl_dir, "*.txt"))
    if not labels:
        print(f"[ERREUR] Aucun fichier .txt dans '{lbl_dir}'.")
        sys.exit(1)

    date_tag = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    target_dir = os.path.join(os.path.dirname(out_dir), "annotations", f"labels_preannotated_{date_tag}")
    os.makedirs(target_dir, exist_ok=True)

    copied = 0
    for l in labels:
        shutil.copy2(l, target_dir)
        copied += 1

    print("\n" + "=" * 65)
    print("  LABELS APPROUVÉS ET EXPORTÉS")
    print(f"  Fichiers copiés     : {copied}")
    print(f"  Dossier destination : {target_dir}")
    print("-" * 65)
    print("  Vous pouvez relancer l'entraînement maintenant avec :")
    print("  python oak_tools/train_yolo.py")
    print("=" * 65 + "\n")


# ==============================================================================
# 7. Pipeline Principal de Pré-Annotation
# ==============================================================================
def parse_args():
    raw_default, weights_default, out_default = resolve_default_paths()

    parser = argparse.ArgumentParser(
        description="Pré-annote automatiquement les images d'un dataset avec YOLOv8 et vérifie visuellement les détections"
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default=raw_default,
        help=f"Dossier source des images brutes (Défaut: {raw_default})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=out_default,
        help=f"Dossier de sortie des pré-annotations (Défaut: {out_default})",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default=weights_default,
        help=f"Poids du modèle YOLO (Défaut: {weights_default})",
    )
    parser.add_argument(
        "--conf",
        "-c",
        type=float,
        default=0.25,
        help="Seuil de confiance minimum pour retenir une détection (0.05 à 0.90, Défaut: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Seuil NMS IoU (Défaut: 0.45)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Taille de redimensionnement pour l'inférence (Défaut: 960)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        choices=["all", "oak", "wrist"],
        default="all",
        help="Filtrer par type de caméra ('all', 'oak', 'wrist'). Défaut: all",
    )
    parser.add_argument(
        "--view",
        "-v",
        action="store_true",
        help="Lancer le visualiseur interactif OpenCV après la pré-annotation",
    )
    parser.add_argument(
        "--view-only",
        action="store_true",
        help="Lancer uniquement le visualiseur interactif sur le dossier de sortie existant sans recalculer",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Valider et copier les labels du dossier de sortie directement dans annotations/ pour l'entraînement",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        default=True,
        help="Enregistrer les découpes (crops) des tubes et racks détectés (Défaut: True)",
    )
    parser.add_argument(
        "--no-crops",
        dest="save_crops",
        action="store_false",
        help="Ne pas sauvegarder les crops individuels",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Si l'utilisateur demande juste d'approuver
    if args.approve:
        approve_and_export(args.output_dir)
        return

    # Si l'utilisateur demande juste de visualiser un dossier existant
    if args.view_only:
        summary_path = os.path.join(args.output_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            metadata_list = data.get("images", [])
        else:
            vis_files = sorted(glob.glob(os.path.join(args.output_dir, "visualized", "*.jpg")))
            metadata_list = [{"filename": os.path.basename(f), "num_detections": "?", "num_tubes": "?", "num_racks": "?"} for f in vis_files]
        run_interactive_viewer(args.output_dir, metadata_list)
        return

    if not args.input_dir or not os.path.exists(args.input_dir):
        print(f"[ERREUR] Le dossier source '{args.input_dir}' n'existe pas.")
        sys.exit(1)

    if not args.weights or not os.path.exists(args.weights):
        print(f"[ERREUR] Les poids '{args.weights}' sont introuvables.")
        sys.exit(1)

    # Récupérer les images
    input_path = os.path.abspath(args.input_dir)
    img_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    all_images = []
    for ext in img_extensions:
        all_images.extend(glob.glob(os.path.join(input_path, ext)))
        if os.path.isdir(os.path.join(input_path, "all")):
            all_images.extend(glob.glob(os.path.join(input_path, "all", ext)))

    all_images = sorted(list(set(all_images)))

    if args.camera == "oak":
        all_images = [f for f in all_images if "oak_" in os.path.basename(f)]
    elif args.camera == "wrist":
        all_images = [f for f in all_images if "wrist_" in os.path.basename(f)]

    if not all_images:
        print(f"[ERREUR] Aucune image trouvée dans '{input_path}'.")
        sys.exit(1)

    # Préparation des dossiers de sortie
    out_dir = os.path.abspath(args.output_dir)
    img_dir = os.path.join(out_dir, "images")
    lbl_dir = os.path.join(out_dir, "labels")
    vis_dir = os.path.join(out_dir, "visualized")
    crops_dir = os.path.join(out_dir, "crops")
    for d in [img_dir, lbl_dir, vis_dir]:
        os.makedirs(d, exist_ok=True)
    if args.save_crops:
        os.makedirs(crops_dir, exist_ok=True)

    print("=" * 70)
    print("  ROBOTKRAFT - PRÉ-ANNOTATION AUTOMATIQUE DU DATASET")
    print(f"  Images sources      : {input_path} ({len(all_images)} images)")
    print(f"  Modèle YOLO         : {args.weights}")
    print(f"  Seuil de confiance  : {args.conf}")
    print(f"  Dossier de sortie   : {out_dir}")
    print("=" * 70)

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Chargement du modèle sur device='{device}'...")
    model = YOLO(args.weights)
    class_names = model.names
    print(f"[INFO] Classes détectables : {class_names}")

    RACK_COLOR = (220, 50, 200)

    stats = {
        "weights": args.weights,
        "conf": args.conf,
        "total_images": len(all_images),
        "total_detections": 0,
        "total_tubes": 0,
        "total_racks": 0,
        "empty_images": 0,
        "empty_files": [],
        "oak_images": 0,
        "wrist_images": 0,
        "tube_colors_summary": {"JAUNE": 0, "CYAN": 0, "MAGENTA": 0, "TRANSPARENT": 0, "INCONNU": 0},
    }

    # Initialisation pour MakeSense.ai (format COCO JSON en fichier unique)
    coco_categories = [{"id": cid + 1, "name": class_names[cid]} for cid in sorted(class_names.keys())]
    coco_data = {
        "info": {
            "description": "RobotKraft Pre-annotations for MakeSense.ai",
            "date_created": datetime.now().isoformat(),
        },
        "images": [],
        "categories": coco_categories,
        "annotations": [],
    }
    coco_ann_id = 1

    metadata_list = []
    t0 = time.time()

    print("\n[INFO] Démarrage de l'inférence et de l'annotation...")
    for idx, img_path in enumerate(all_images, 1):
        fname = os.path.basename(img_path)
        base_name = os.path.splitext(fname)[0]
        cam_type = "OAK" if fname.startswith("oak_") else ("Wrist" if fname.startswith("wrist_") else "Camera")

        if cam_type == "OAK":
            stats["oak_images"] += 1
        elif cam_type == "Wrist":
            stats["wrist_images"] += 1

        # Lien symbolique vers l'image source dans images/
        symlink_target = os.path.join(img_dir, fname)
        if not os.path.exists(symlink_target):
            try:
                os.symlink(img_path, symlink_target)
            except OSError:
                pass

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [{idx}/{len(all_images)}] [ATTENTION] Impossible de lire {fname}")
            continue

        orig_h, orig_w = img.shape[:2]
        annotated_img = img.copy()

        # Enregistrement de l'image dans le format COCO
        coco_data["images"].append({
            "id": idx,
            "file_name": fname,
            "width": orig_w,
            "height": orig_h,
        })

        # Inférence YOLO
        results = model.predict(
            source=img,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
        )[0]

        boxes = results.boxes
        num_dets = len(boxes)
        stats["total_detections"] += num_dets
        if num_dets == 0:
            stats["empty_images"] += 1
            stats["empty_files"].append(fname)

        tubes_cnt = 0
        racks_cnt = 0
        tube_colors = {}
        yolo_lines = []

        if num_dets > 0:
            for b in boxes:
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                cls_name = class_names.get(cls_id, str(cls_id))

                x1, y1, x2, y2 = map(int, b.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)

                # Format YOLO normalisé (xc, yc, w, h)
                xc = ((x1 + x2) / 2.0) / orig_w
                yc = ((y1 + y2) / 2.0) / orig_h
                bw = (x2 - x1) / float(orig_w)
                bh = (y2 - y1) / float(orig_h)
                yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

                # Format COCO pour MakeSense (bbox: [x_min, y_min, width, height] en pixels)
                box_w = x2 - x1
                box_h = y2 - y1
                coco_data["annotations"].append({
                    "id": coco_ann_id,
                    "image_id": idx,
                    "category_id": cls_id + 1,
                    "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(box_w), 2), round(float(box_h), 2)],
                    "area": round(float(box_w * box_h), 2),
                    "segmentation": [],
                    "iscrowd": 0,
                })
                coco_ann_id += 1

                crop = img[y1:y2, x1:x2]
                if cls_name == "tube":
                    tubes_cnt += 1
                    stats["total_tubes"] += 1
                    col_name, col_bgr = classify_liquid_color(crop)
                    tube_colors[col_name] = tube_colors.get(col_name, 0) + 1
                    stats["tube_colors_summary"][col_name] = stats["tube_colors_summary"].get(col_name, 0) + 1

                    label_text = f"tube: {col_name} {int(conf * 100)}%"
                    draw_detection_box(annotated_img, x1, y1, x2, y2, label_text, col_bgr)

                    if args.save_crops and crop.size > 0:
                        crop_sub = os.path.join(crops_dir, "tube", col_name)
                        os.makedirs(crop_sub, exist_ok=True)
                        crop_name = f"{base_name}_tube_{tubes_cnt}_{col_name}_{int(conf*100)}.jpg"
                        cv2.imwrite(os.path.join(crop_sub, crop_name), crop)

                elif cls_name == "rack":
                    racks_cnt += 1
                    stats["total_racks"] += 1
                    label_text = f"rack {int(conf * 100)}%"
                    draw_detection_box(annotated_img, x1, y1, x2, y2, label_text, RACK_COLOR, text_color=(255, 255, 255))

                    if args.save_crops and crop.size > 0:
                        crop_sub = os.path.join(crops_dir, "rack")
                        os.makedirs(crop_sub, exist_ok=True)
                        crop_name = f"{base_name}_rack_{racks_cnt}_{int(conf*100)}.jpg"
                        cv2.imwrite(os.path.join(crop_sub, crop_name), crop)
                else:
                    label_text = f"{cls_name} {int(conf * 100)}%"
                    draw_detection_box(annotated_img, x1, y1, x2, y2, label_text, (0, 255, 0))

        # Écriture du fichier d'annotation YOLO (.txt)
        txt_path = os.path.join(lbl_dir, f"{base_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines) + "\n")

        # Sauvegarde de l'image visualisée
        vis_path = os.path.join(vis_dir, fname)
        cv2.imwrite(vis_path, annotated_img)

        metadata_list.append({
            "filename": fname,
            "camera": cam_type,
            "num_detections": num_dets,
            "num_tubes": tubes_cnt,
            "num_racks": racks_cnt,
            "tube_colors": tube_colors,
        })

        if idx % 10 == 0 or idx == len(all_images):
            print(f"  [{idx:02d}/{len(all_images)}] Traitée : {fname} -> {num_dets} détections ({tubes_cnt} tubes, {racks_cnt} racks)")

    elapsed = time.time() - t0
    stats["elapsed_seconds"] = round(elapsed, 2)
    stats["fps"] = round(len(all_images) / elapsed, 1) if elapsed > 0 else 0

    # Sauvegarde du fichier unique MakeSense (format COCO JSON)
    makesense_json_path = os.path.join(out_dir, "makesense_annotations.json")
    with open(makesense_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)

    # Sauvegarde du labels.txt (utile si import YOLO multi-fichiers dans MakeSense)
    labels_txt_path = os.path.join(lbl_dir, "labels.txt")
    with open(labels_txt_path, "w", encoding="utf-8") as f:
        for cid in sorted(class_names.keys()):
            f.write(f"{class_names[cid]}\n")

    # Sauvegarde du summary.json
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "images": metadata_list}, f, indent=2, ensure_ascii=False)

    # Génération du rapport HTML
    html_path = generate_html_report(out_dir, metadata_list, stats)

    print("\n" + "=" * 70)
    print("  PRÉ-ANNOTATION TERMINÉE AVEC SUCCÈS")
    print(f"  Temps d'exécution    : {elapsed:.2f}s ({stats['fps']} img/s)")
    print(f"  Images traitées      : {stats['total_images']} (OAK: {stats['oak_images']}, Wrist: {stats['wrist_images']})")
    print(f"  Total détections     : {stats['total_detections']}")
    print(f"    - Tubes            : {stats['total_tubes']}")
    for col, count in stats["tube_colors_summary"].items():
        if count > 0:
            print(f"        * {col:11s}: {count}")
    print(f"    - Racks            : {stats['total_racks']}")
    print(f"    - Sans détection   : {stats['empty_images']} image(s)")
    print("-" * 70)
    print(f"  FICHIER UNIQUE MAKESENSE : {makesense_json_path}")
    print(f"  Dossier des images       : {img_dir}")
    print(f"  Dossier des labels YOLO  : {lbl_dir}")
    print(f"  Dossier des visuels      : {vis_dir}")
    print(f"  Rapport HTML interactif  : {html_path}")
    print(f"  Résumé statistiques      : {summary_path}")
    print("=" * 70)

    print("\n[IMPORT DANS MAKESENSE.AI] :")
    print("  1. Allez sur https://www.makesense.ai/ et glissez vos images brutes.")
    print("  2. Cliquez sur 'Object Detection' puis 'Rect'.")
    print("  3. Menu haut gauche : 'Actions' -> 'Import annotations'.")
    print("  4. Sélectionnez l'option : 'Single file in COCO JSON format'.")
    print(f"  5. Glissez le fichier unique :")
    print(f"     {makesense_json_path}")

    print("\n[ASTUCE] Pour vérifier visuellement dans le navigateur :")
    print(f"  xdg-open {html_path}")
    print(f"\n[ASTUCE] Pour valider et exporter directement vers annotations/ :")
    print(f"  python {sys.argv[0]} --approve")
    print(f"  python oak_tools/train_yolo.py\n")

    if args.view:
        run_interactive_viewer(out_dir, metadata_list)


if __name__ == "__main__":
    main()
