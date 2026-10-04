#!/usr/bin/env python3
"""Entraîne une policy LeRobot sur un PC distant à partir d'un dataset local, sans passer par le Hub.

Équivalent de `make train` / notebooks/train_act.ipynb, mais le dataset est copié à la main
(parquet + vidéos caméra) au lieu d'être téléchargé depuis Hugging Face.

1. Côté robot, archiver le dataset enregistré par `make record` (les fichiers appartiennent à root,
   un zip sans sudo perd les .mp4 sans prévenir) :
       cd .cache/huggingface/lerobot/<HF_USER>
       sudo zip -r <TASK>.zip <TASK> && sudo chown $USER <TASK>.zip
2. Copier l'archive et ce script sur le PC distant (scp, clé USB...).
3. Sur le PC distant (Python 3.12, GPU NVIDIA) :
       pip install "lerobot==0.5.1"
       python train_remote.py <TASK>.zip --steps 30000
4. Rapatrier l'archive <TASK>_pretrained_model.zip produite, puis côté robot :
       mkdir -p outputs/<TASK>/checkpoints/last/pretrained_model
       unzip <TASK>_pretrained_model.zip -d outputs/<TASK>/checkpoints/last/pretrained_model
       make eval TASK=<TASK>

Reprendre un run interrompu : relancer la même commande avec --resume.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

LEROBOT_VERSION = "0.5.1"  # même version que le conteneur (pyproject.toml)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path, help="Dossier du dataset (meta/, data/, videos/) ou archive .zip")
    p.add_argument("--task", help="Nom de la tâche (défaut : nom du dossier du dataset)")
    p.add_argument("--policy", default="act", help="Type de policy LeRobot (défaut : act)")
    p.add_argument("--steps", type=int, default=30_000, help="20k-50k suffisent souvent pour un petit dataset")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--save-freq", type=int, default=10_000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", choices=["cuda", "mps", "cpu"], help="Défaut : détecté automatiquement")
    p.add_argument("--output-dir", type=Path, help="Défaut : outputs/<TASK>")
    p.add_argument("--resume", action="store_true", help="Reprend depuis <output-dir>/checkpoints/last")
    p.add_argument("--extract-dir", type=Path, default=Path("datasets_local"), help="Où décompresser un .zip")
    p.add_argument("--dry-run", action="store_true", help="Vérifie le dataset et affiche la commande sans lancer")
    p.epilog = "Toute autre option (ex. --policy.chunk_size=50) est passée telle quelle à lerobot-train."
    args, args.extra = p.parse_known_args()
    return args


def extract_zip(archive: Path, dest: Path) -> Path:
    """Décompresse l'archive et renvoie le dossier qui contient meta/info.json."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        infos = [n for n in zf.namelist() if n.endswith("meta/info.json")]
        if len(infos) != 1:
            sys.exit(f"{archive} doit contenir exactement un meta/info.json (trouvé : {len(infos)})")
        zf.extractall(dest)
    return (dest / infos[0]).parent.parent


def check_dataset(root: Path) -> dict:
    """Vérifie que parquet et vidéos sont présents et lisibles, renvoie info.json."""
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        sys.exit(f"{info_path} introuvable : le dossier doit contenir meta/, data/ et videos/")
    info = json.loads(info_path.read_text())

    if info.get("codebase_version") != "v3.0":
        print(f"⚠ codebase_version={info.get('codebase_version')}, attendu v3.0 (lerobot {LEROBOT_VERSION})")

    errors = []
    for rel in ("meta/stats.json", "meta/tasks.parquet"):
        if not (root / rel).is_file():
            errors.append(f"{rel} manquant")
    if not list((root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        errors.append("meta/episodes/chunk-*/*.parquet manquant")
    if not list((root / "data").glob("chunk-*/*.parquet")):
        errors.append("data/chunk-*/*.parquet manquant")

    cameras = [k for k, f in info["features"].items() if f["dtype"] in ("video", "image")]
    for cam in cameras:
        if info["features"][cam]["dtype"] != "video":
            continue
        videos = list((root / "videos" / cam).glob("chunk-*/*.mp4"))
        if not videos:
            errors.append(f"aucune vidéo pour {cam} (zip créé sans sudo ? les .mp4 appartiennent à root)")
        errors += [f"{v} illisible (droits)" for v in videos if not os.access(v, os.R_OK)]

    if errors:
        sys.exit("Dataset incomplet :\n  - " + "\n  - ".join(errors))
    if not cameras:
        print("⚠ aucune caméra dans le dataset : ACT requiert au moins une observation.images.*")

    print(f"Dataset : {root}")
    print(f"  {info['total_episodes']} épisodes, {info['total_frames']} frames à {info['fps']} fps")
    for cam in cameras:
        h, w, _ = info["features"][cam]["shape"]
        print(f"  caméra {cam} : {w}x{h}")
    return info


def detect_device() -> str:
    try:
        import torch
    except ImportError:
        sys.exit(f'torch introuvable : pip install "lerobot=={LEROBOT_VERSION}"')
    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    print("⚠ pas de GPU visible, entraînement sur CPU (très lent)")
    return "cpu"


def check_lerobot():
    try:
        import lerobot
    except ImportError:
        sys.exit(f'lerobot introuvable : pip install "lerobot=={LEROBOT_VERSION}"')
    if lerobot.__version__ != LEROBOT_VERSION:
        print(f"⚠ lerobot {lerobot.__version__} installé, le robot utilise {LEROBOT_VERSION} : "
              "le checkpoint risque de ne pas se recharger en eval")
    if not shutil.which("lerobot-train"):
        sys.exit("commande lerobot-train introuvable dans le PATH (venv activé ?)")


def main():
    args = parse_args()
    check_lerobot()

    src = args.dataset.expanduser().resolve()
    if src.is_file() and src.suffix == ".zip":
        root = extract_zip(src, args.extract_dir.resolve())
    elif src.is_dir():
        root = src
    else:
        sys.exit(f"{src} n'est ni un dossier ni une archive .zip")

    check_dataset(root)
    task = args.task or root.name
    output_dir = (args.output_dir or Path("outputs") / task).resolve()
    device = args.device or detect_device()

    # Dataset lu uniquement depuis le disque : aucun accès au Hub. Le repo_id n'est qu'un identifiant.
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}

    if args.resume:
        config = output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        if not config.is_file():
            sys.exit(f"{config} introuvable, rien à reprendre")
        cmd = ["lerobot-train", f"--config_path={config}", "--resume=true"]
    else:
        if output_dir.exists():
            sys.exit(f"{output_dir} existe déjà : --resume pour reprendre, ou --output-dir pour un nouveau run")
        cmd = [
            "lerobot-train",
            f"--policy.type={args.policy}",
            f"--policy.device={device}",
            "--policy.push_to_hub=false",
            f"--dataset.repo_id=local/{task}",
            f"--dataset.root={root}",
            f"--output_dir={output_dir}",
            f"--job_name={args.policy}_{task}",
            f"--steps={args.steps}",
            f"--batch_size={args.batch_size}",
            f"--save_freq={args.save_freq}",
            f"--num_workers={args.num_workers}",
            "--wandb.enable=false",
        ]
    cmd += args.extra

    print("\n" + " \\\n  ".join(cmd) + "\n")
    if args.dry_run:
        return
    subprocess.run(cmd, env=env, check=True)

    ckpt = output_dir / "checkpoints" / "last" / "pretrained_model"
    archive = shutil.make_archive(str(output_dir.parent / f"{task}_pretrained_model"), "zip", ckpt)
    print(f"\nModèle : {ckpt}\nArchive à rapatrier côté robot : {archive}")
    print(f"  unzip {Path(archive).name} -d outputs/{task}/checkpoints/last/pretrained_model")
    print(f"  make eval TASK={task}")


if __name__ == "__main__":
    main()
