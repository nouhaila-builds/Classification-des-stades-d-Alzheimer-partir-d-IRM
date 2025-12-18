"""
Script principal pour la détection précoce de la maladie d'Alzheimer
via Deep Learning
"""

import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from typing import Dict, List

from data_preprocessing import MedicalImagePreprocessor
from models.cnn_models import SimpleCNN, AdvancedCNN, MultiScaleCNN, compile_model
from models.transfer_learning import TransferLearningModel, compile_transfer_model
from models.rnn_models import LSTMClassifier, GRUClassifier, BidirectionalLSTM, compile_rnn_model
from models.segmentation_models import UNet, SegNet, AttentionUNet, compile_segmentation_model
from training import ModelTrainer, calculate_class_weights, train_multiple_models
from evaluation import ModelEvaluator
from utils import (
    create_directories, save_results, get_class_names,
    print_model_summary, set_seed
)


def load_sample_data(num_samples: int = 1000, image_shape: tuple = (224, 224, 1)):
    """
    Charge des données d'exemple pour la démonstration
    
    Args:
        num_samples: Nombre d'échantillons
        image_shape: Forme des images
        
    Returns:
        Tuple (X, y) - données et labels
    """
    print(f"Génération de {num_samples} échantillons d'exemple...")
    
    X = np.random.rand(num_samples, *image_shape).astype(np.float32)
    y = np.random.randint(0, 4, size=num_samples)
    
    return X, y


def train_classification_models(X_train, y_train, X_val, y_val, input_shape):
    """
    Entraîne différents modèles de classification
    
    Args:
        X_train: Données d'entraînement
        y_train: Labels d'entraînement
        X_val: Données de validation
        y_val: Labels de validation
        input_shape: Forme des images d'entrée
        
    Returns:
        Dictionnaire des résultats
    """
    results = {}
    num_classes = len(np.unique(y_train))
    
    models_config = [
        {
            'name': 'SimpleCNN',
            'model': compile_model(SimpleCNN.build(input_shape, num_classes))
        },
        {
            'name': 'AdvancedCNN',
            'model': compile_model(AdvancedCNN.build(input_shape, num_classes))
        },
        {
            'name': 'MultiScaleCNN',
            'model': compile_model(MultiScaleCNN.build(input_shape, num_classes))
        },
        {
            'name': 'VGG16',
            'model': compile_transfer_model(
                TransferLearningModel.build_vgg16(input_shape, num_classes)
            )
        },
        {
            'name': 'ResNet50',
            'model': compile_transfer_model(
                TransferLearningModel.build_resnet50(input_shape, num_classes)
            )
        }
    ]
    
    class_weights = calculate_class_weights(y_train)
    
    for config in models_config:
        model_name = config['name']
        model = config['model']
        
        print(f"\n{'='*60}")
        print(f"Entraînement de {model_name}")
        print(f"{'='*60}")
        
        print_model_summary(model)
        
        trainer = ModelTrainer(model, model_name)
        history = trainer.train(
            X_train, y_train, X_val, y_val,
            batch_size=32,
            epochs=50,
            class_weights=class_weights,
            use_data_augmentation=True
        )
        
        trainer.save_model()
        
        y_pred = model.predict(X_val).argmax(axis=1)
        y_pred_proba = model.predict(X_val)
        
        evaluator = ModelEvaluator(get_class_names())
        metrics = evaluator.calculate_metrics(y_val, y_pred, y_pred_proba)
        
        results[model_name] = {
            'history': history,
            'metrics': metrics,
            'model': model
        }
        
        print(f"\nMétriques pour {model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    return results


def train_segmentation_models(X_train, y_train, X_val, y_val, input_shape):
    """
    Entraîne des modèles de segmentation
    
    Args:
        X_train: Données d'entraînement
        y_train: Masques d'entraînement
        X_val: Données de validation
        y_val: Masques de validation
        input_shape: Forme des images d'entrée
        
    Returns:
        Dictionnaire des résultats
    """
    results = {}
    
    segmentation_models = [
        ('UNet', UNet),
        ('SegNet', SegNet),
        ('AttentionUNet', AttentionUNet)
    ]
    
    for model_name, model_class in segmentation_models:
        print(f"\n{'='*60}")
        print(f"Entraînement de {model_name}")
        print(f"{'='*60}")
        
        model = compile_segmentation_model(
            model_class.build(input_shape, num_classes=1),
            num_classes=1
        )
        
        print_model_summary(model)
        
        trainer = ModelTrainer(model, f'{model_name}_segmentation')
        history = trainer.train(
            X_train, y_train, X_val, y_val,
            batch_size=16,
            epochs=30
        )
        
        trainer.save_model()
        
        y_pred = (model.predict(X_val) > 0.5).astype(np.uint8)
        
        evaluator = ModelEvaluator()
        seg_metrics = evaluator.evaluate_segmentation(y_val, y_pred)
        
        results[model_name] = {
            'history': history,
            'metrics': seg_metrics,
            'model': model
        }
        
        print(f"\nMétriques de segmentation pour {model_name}:")
        for metric, value in seg_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    return results


def compare_all_models(results: Dict):
    """
    Compare tous les modèles entraînés
    
    Args:
        results: Dictionnaire des résultats
    """
    print("\n" + "="*60)
    print("COMPARAISON DES MODÈLES")
    print("="*60)
    
    metrics_dict = {}
    for model_name, result in results.items():
        if 'metrics' in result:
            metrics_dict[model_name] = result['metrics']
    
    evaluator = ModelEvaluator(get_class_names())
    evaluator.compare_models(metrics_dict, save_path='results/figures/model_comparison.png')
    
    print("\nRésumé des performances:")
    print("-" * 60)
    print(f"{'Modèle':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)
    
    for model_name, metrics in metrics_dict.items():
        print(f"{model_name:<20} "
              f"{metrics.get('accuracy', 0):<12.4f} "
              f"{metrics.get('precision', 0):<12.4f} "
              f"{metrics.get('recall', 0):<12.4f} "
              f"{metrics.get('f1_score', 0):<12.4f}")


def main():
    """
    Fonction principale
    """
    parser = argparse.ArgumentParser(description='Détection précoce de la maladie d\'Alzheimer')
    parser.add_argument('--mode', type=str, default='classification',
                       choices=['classification', 'segmentation', 'both'],
                       help='Mode d\'entraînement')
    parser.add_argument('--seed', type=int, default=42, help='Graine aléatoire')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Répertoire des données')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    create_directories()
    
    print("="*60)
    print("DÉTECTION PRÉCOCE DE LA MALADIE D'ALZHEIMER")
    print("via Deep Learning")
    print("="*60)
    
    input_shape = (224, 224, 1)
    
    print("\nChargement des données...")
    X, y = load_sample_data(num_samples=1000, image_shape=input_shape)
    
    print("\nPrétraitement des données...")
    preprocessor = MedicalImagePreprocessor(target_size=(224, 224))
    X_processed, y_processed = preprocessor.preprocess_batch(
        [X[i] for i in range(len(X))],
        y.tolist()
    )
    
    X_balanced, y_balanced = preprocessor.handle_class_imbalance(
        X_processed, y_processed, strategy='smote'
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\nDonnées d'entraînement: {X_train.shape}")
    print(f"Données de validation: {X_val.shape}")
    print(f"Données de test: {X_test.shape}")
    
    all_results = {}
    
    if args.mode in ['classification', 'both']:
        print("\n" + "="*60)
        print("ENTRAÎNEMENT DES MODÈLES DE CLASSIFICATION")
        print("="*60)
        
        classification_results = train_classification_models(
            X_train, y_train, X_val, y_val, input_shape
        )
        all_results.update(classification_results)
    
    if args.mode in ['segmentation', 'both']:
        print("\n" + "="*60)
        print("ENTRAÎNEMENT DES MODÈLES DE SEGMENTATION")
        print("="*60)
        
        y_train_seg = (np.random.rand(*X_train.shape[:3]) > 0.5).astype(np.float32)
        y_val_seg = (np.random.rand(*X_val.shape[:3]) > 0.5).astype(np.float32)
        
        segmentation_results = train_segmentation_models(
            X_train, y_train_seg, X_val, y_val_seg, input_shape
        )
        all_results.update(segmentation_results)
    
    if all_results:
        compare_all_models(all_results)
        save_results(all_results, 'results/reports/all_results.json')
        
        print("\n" + "="*60)
        print("ENTRAÎNEMENT TERMINÉ")
        print("="*60)
        print("\nRésultats sauvegardés dans:")
        print("  - models/saved_models/ (modèles)")
        print("  - results/reports/all_results.json (métriques)")
        print("  - results/figures/ (graphiques)")
        print("  - logs/ (TensorBoard)")
    
    print("\nPour visualiser les résultats avec TensorBoard:")
    print("  tensorboard --logdir=logs/")


if __name__ == '__main__':
    main()

