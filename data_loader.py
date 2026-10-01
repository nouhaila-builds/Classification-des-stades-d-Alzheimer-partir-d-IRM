"""
Chargement du jeu Kaggle Alzheimer's Multiclass Dataset
(aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented).

Les images restent sur disque et sont lues par lots. Le découpage
train / validation / test est fait avant tout entraînement.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATASET_ID = "aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented"

# Ordre clinique, du sujet sans démence au stade modéré.
CLASS_NAMES = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def download_dataset() -> str:
    """Télécharge la dernière version du jeu et retourne son dossier local."""
    import kagglehub

    path = kagglehub.dataset_download(DATASET_ID)
    print(f"Jeu de données disponible dans: {path}")
    return path


def _normalize_class_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def find_image_root(dataset_path: str) -> str:
    """
    Trouve le dossier qui contient les quatre classes.

    kagglehub place parfois les images dans un sous-dossier
    (combined_images) plutôt qu'à la racine du cache.
    """
    expected = {_normalize_class_name(name) for name in CLASS_NAMES}

    for dirpath, dirnames, _ in os.walk(dataset_path):
        present = {_normalize_class_name(name) for name in dirnames}
        if expected.issubset(present):
            return dirpath

    raise FileNotFoundError(
        "Les dossiers de classes "
        f"{CLASS_NAMES} sont introuvables sous {dataset_path}."
    )


def _class_directories(image_root: str) -> Dict[str, str]:
    by_key = {}
    for name in os.listdir(image_root):
        path = os.path.join(image_root, name)
        if os.path.isdir(path):
            by_key[_normalize_class_name(name)] = path

    directories = {}
    for class_name in CLASS_NAMES:
        key = _normalize_class_name(class_name)
        if key not in by_key:
            raise FileNotFoundError(
                f"Dossier manquant pour la classe {class_name} dans {image_root}."
            )
        directories[class_name] = by_key[key]
    return directories


def collect_samples(
    image_root: str,
    max_per_class: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Liste les chemins d'images et leurs labels.

    max_per_class sert uniquement à un essai rapide. L'échantillon est
    tiré au hasard, avec la graine fournie.
    """
    paths: List[str] = []
    labels: List[int] = []
    rng = np.random.default_rng(seed)

    for class_index, class_name in enumerate(CLASS_NAMES):
        folder = _class_directories(image_root)[class_name]
        files = [
            os.path.join(folder, filename)
            for filename in os.listdir(folder)
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
        ]
        files.sort()
        if not files:
            raise FileNotFoundError(f"Aucune image dans {folder}.")

        if max_per_class is not None and max_per_class < len(files):
            chosen = rng.choice(len(files), size=max_per_class, replace=False)
            files = [files[int(i)] for i in sorted(chosen)]

        paths.extend(files)
        labels.extend([class_index] * len(files))
        print(f"  {class_name}: {len(files)} images")

    return np.array(paths), np.array(labels, dtype=np.int32)


def split_samples(
    paths: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Découpe stratifiée train / validation / test.

    val_size et test_size sont des fractions du jeu complet.
    """
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size doit rester inférieur à 1.")

    paths_train, paths_test, y_train, y_test = train_test_split(
        paths,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    relative_val = val_size / (1.0 - test_size)
    paths_train, paths_val, y_train, y_val = train_test_split(
        paths_train,
        y_train,
        test_size=relative_val,
        random_state=seed,
        stratify=y_train,
    )

    splits = {
        "train": (paths_train, y_train),
        "val": (paths_val, y_val),
        "test": (paths_test, y_test),
    }
    for name, (split_paths, split_labels) in splits.items():
        counts = np.bincount(split_labels, minlength=len(CLASS_NAMES))
        detail = ", ".join(
            f"{class_name}={int(counts[i])}" for i, class_name in enumerate(CLASS_NAMES)
        )
        print(f"{name}: {len(split_paths)} images ({detail})")
    return splits


def _load_image(path: tf.Tensor, label: tf.Tensor, image_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_dataset(
    paths: Sequence[str],
    labels: Sequence[int],
    image_size: int = 224,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Construit un tf.data qui lit, redimensionne et normalise les images."""
    dataset = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def _map(path, label):
        return _load_image(path, label, image_size)

    return (
        dataset
        .map(_map, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
