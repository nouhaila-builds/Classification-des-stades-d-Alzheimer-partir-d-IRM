"""
Utilitaires pour le projet de détection d'Alzheimer
"""

import numpy as np
import os
import json
from typing import Dict, List, Optional, Tuple
import pickle


def create_directories(base_dir: str = '.'):
    """
    Crée la structure de dossiers nécessaire
    
    Args:
        base_dir: Répertoire de base
    """
    directories = [
        'data/raw',
        'data/processed',
        'models/saved_models',
        'results/figures',
        'results/reports',
        'logs',
        'checkpoints'
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(base_dir, directory), exist_ok=True)
    print("Structure de dossiers créée avec succès")


def save_results(results: Dict, filepath: str):
    """
    Sauvegarde les résultats dans un fichier JSON
    
    Args:
        results: Dictionnaire de résultats
        filepath: Chemin de sauvegarde
    """
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    print(f"Résultats sauvegardés: {filepath}")


def load_results(filepath: str) -> Dict:
    """
    Charge les résultats depuis un fichier JSON
    
    Args:
        filepath: Chemin du fichier
        
    Returns:
        Dictionnaire de résultats
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results


def save_predictions(y_pred: np.ndarray, y_pred_proba: np.ndarray,
                    filepath: str):
    """
    Sauvegarde les prédictions
    
    Args:
        y_pred: Prédictions de classes
        y_pred_proba: Probabilités de prédiction
        filepath: Chemin de sauvegarde
    """
    predictions = {
        'y_pred': y_pred.tolist(),
        'y_pred_proba': y_pred_proba.tolist()
    }
    
    with open(filepath, 'w') as f:
        json.dump(predictions, f, indent=4)
    print(f"Prédictions sauvegardées: {filepath}")


def get_class_mapping() -> Dict[int, str]:
    """
    Retourne le mapping des classes
    
    Returns:
        Dictionnaire {index: nom_classe}
    """
    return {
        0: 'CN',
        1: 'MCI',
        2: 'AD',
        3: 'EMCI'
    }


def get_class_names() -> List[str]:
    """
    Retourne la liste des noms de classes
    
    Returns:
        Liste des noms de classes
    """
    return ['CN', 'MCI', 'AD', 'EMCI']


def print_model_summary(model):
    """
    Affiche un résumé du modèle
    
    Args:
        model: Modèle Keras
    """
    print("\n" + "="*50)
    print("Résumé du modèle")
    print("="*50)
    model.summary()
    print(f"\nNombre total de paramètres: {model.count_params():,}")


def set_seed(seed: int = 42):
    """
    Définit la graine aléatoire pour la reproductibilité
    
    Args:
        seed: Valeur de la graine
    """
    np.random.seed(seed)
    import random
    random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Graine aléatoire définie à {seed}")


def format_time(seconds: float) -> str:
    """
    Formate le temps en secondes en format lisible
    
    Args:
        seconds: Temps en secondes
        
    Returns:
        String formatée
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

