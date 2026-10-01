"""
Module d'entraînement pour les modèles de détection d'Alzheimer
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, LearningRateScheduler
)
from typing import Tuple, Optional, Dict, List
import os
from datetime import datetime


class ModelTrainer:
    """Classe pour l'entraînement des modèles"""
    
    def __init__(self, model: keras.Model, model_name: str = 'model',
                 save_dir: str = 'models/saved_models'):
        """
        Initialise le trainer
        
        Args:
            model: Modèle Keras à entraîner
            model_name: Nom du modèle
            save_dir: Répertoire pour sauvegarder les modèles
        """
        self.model = model
        self.model_name = model_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.history = None
    
    def create_callbacks(self, monitor: str = 'val_accuracy',
                        patience: int = 15, min_lr: float = 1e-7) -> List:
        """
        Crée les callbacks pour l'entraînement
        
        Args:
            monitor: Métrique à surveiller
            patience: Patience pour early stopping
            min_lr: Taux d'apprentissage minimum
            
        Returns:
            Liste de callbacks
        """
        callbacks = []
        
        model_path = os.path.join(self.save_dir, f'{self.model_name}_best.h5')
        checkpoint = ModelCheckpoint(
            model_path,
            monitor=monitor,
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        reduce_lr = ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=5,
            min_lr=min_lr,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        log_dir = os.path.join('logs', self.model_name, datetime.now().strftime('%Y%m%d-%H%M%S'))
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0)
        callbacks.append(tensorboard)
        
        return callbacks
    
    def train(self, X_train, y_train: Optional[np.ndarray] = None,
             X_val=None, y_val: Optional[np.ndarray] = None,
             batch_size: int = 32, epochs: int = 100,
             class_weights: Optional[Dict] = None,
             use_data_augmentation: bool = False,
             patience: int = 15) -> Dict:
        """
        Entraîne le modèle
        
        Args:
            X_train: Données d'entraînement
            y_train: Labels d'entraînement
            X_val: Données de validation
            y_val: Labels de validation
            batch_size: Taille du batch
            epochs: Nombre d'époques
            class_weights: Poids des classes pour gérer le déséquilibre
            use_data_augmentation: Si True, utilise l'augmentation de données
            
        Returns:
            Historique d'entraînement
        """
        callbacks = self.create_callbacks(patience=patience)

        if isinstance(X_train, tf.data.Dataset):
            if use_data_augmentation:
                print(
                    "Augmentation en mémoire ignorée: le jeu est déjà augmenté "
                    "et les images sont lues depuis le disque."
                )
            self.history = self.model.fit(
                X_train,
                validation_data=X_val,
                epochs=epochs,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
            return self.history.history
        
        if use_data_augmentation:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            
            datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                zoom_range=0.1
            )
            
            train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
            
            steps_per_epoch = len(X_train) // batch_size
            
            self.history = self.model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                class_weight=class_weights
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
        
        return self.history.history
    
    def fine_tune(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  unfreeze_layers: int = 10,
                  learning_rate: float = 1e-5,
                  batch_size: int = 32, epochs: int = 50) -> Dict:
        """
        Effectue le fine-tuning du modèle
        
        Args:
            X_train: Données d'entraînement
            y_train: Labels d'entraînement
            X_val: Données de validation
            y_val: Labels de validation
            unfreeze_layers: Nombre de couches à dégeler
            learning_rate: Nouveau taux d'apprentissage
            batch_size: Taille du batch
            epochs: Nombre d'époques
            
        Returns:
            Historique d'entraînement
        """
        for layer in self.model.layers[-unfreeze_layers:]:
            layer.trainable = True
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = self.create_callbacks()
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def save_model(self, filepath: Optional[str] = None):
        """
        Sauvegarde le modèle
        
        Args:
            filepath: Chemin de sauvegarde (optionnel)
        """
        if filepath is None:
            filepath = os.path.join(self.save_dir, f'{self.model_name}_final.h5')
        
        self.model.save(filepath)
        print(f"Modèle sauvegardé: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Charge un modèle sauvegardé
        
        Args:
            filepath: Chemin vers le modèle
        """
        self.model = keras.models.load_model(filepath)
        print(f"Modèle chargé: {filepath}")


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calcule les poids des classes pour gérer le déséquilibre
    
    Args:
        y: Labels
        
    Returns:
        Dictionnaire {classe: poids}
    """
    from collections import Counter
    
    class_counts = Counter(y)
    total = len(y)
    n_classes = len(class_counts)
    
    weights = {}
    for class_idx, count in class_counts.items():
        weights[int(class_idx)] = total / (n_classes * count)
    
    return weights


def train_multiple_models(models_config: List[Dict], X_train: np.ndarray,
                         y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                         **train_kwargs) -> Dict[str, Dict]:
    """
    Entraîne plusieurs modèles et compare leurs performances
    
    Args:
        models_config: Liste de configurations de modèles
        X_train: Données d'entraînement
        y_train: Labels d'entraînement
        X_val: Données de validation
        y_val: Labels de validation
        **train_kwargs: Arguments supplémentaires pour l'entraînement
        
    Returns:
        Dictionnaire {nom_modèle: historique}
    """
    results = {}
    
    for config in models_config:
        model = config['model']
        model_name = config['name']
        
        print(f"\nEntraînement de {model_name}...")
        trainer = ModelTrainer(model, model_name)
        history = trainer.train(X_train, y_train, X_val, y_val, **train_kwargs)
        
        results[model_name] = {
            'history': history,
            'model': model,
            'trainer': trainer
        }
    
    return results

