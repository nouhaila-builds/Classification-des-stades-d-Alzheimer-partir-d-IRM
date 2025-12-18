"""
Module d'évaluation et de métriques pour les modèles de détection d'Alzheimer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from typing import Tuple, List, Optional, Dict
import os


class ModelEvaluator:
    """Classe pour l'évaluation complète des modèles"""
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialise l'évaluateur
        
        Args:
            class_names: Noms des classes (CN, MCI, AD, EMCI)
        """
        if class_names is None:
            self.class_names = ['CN', 'MCI', 'AD', 'EMCI']
        else:
            self.class_names = class_names
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Calcule toutes les métriques de classification
        
        Args:
            y_true: Labels réels
            y_pred: Prédictions (classes)
            y_pred_proba: Probabilités de prédiction (optionnel)
            
        Returns:
            Dictionnaire contenant toutes les métriques
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(
                        y_true, y_pred_proba, multi_class='ovr', average='weighted'
                    )
                metrics['average_precision'] = average_precision_score(
                    y_true, y_pred_proba, average='weighted'
                )
            except Exception as e:
                print(f"Erreur lors du calcul de ROC-AUC: {e}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             save_path: Optional[str] = None, title: str = 'Confusion Matrix'):
        """
        Trace la matrice de confusion
        
        Args:
            y_true: Labels réels
            y_pred: Prédictions
            save_path: Chemin pour sauvegarder la figure
            title: Titre de la figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(title)
        plt.ylabel('Vraie classe')
        plt.xlabel('Classe prédite')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      save_path: Optional[str] = None, title: str = 'ROC Curve'):
        """
        Trace la courbe ROC
        
        Args:
            y_true: Labels réels
            y_pred_proba: Probabilités de prédiction
            save_path: Chemin pour sauvegarder la figure
            title: Titre de la figure
        """
        n_classes = len(np.unique(y_true))
        
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Taux de faux positifs')
            plt.ylabel('Taux de vrais positifs')
            plt.title(title)
            plt.legend(loc="lower right")
        else:
            from sklearn.preprocessing import label_binarize
            from itertools import cycle
            
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
            
            plt.figure(figsize=(10, 8))
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
            
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Taux de faux positifs')
            plt.ylabel('Taux de vrais positifs')
            plt.title(title)
            plt.legend(loc="lower right")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   save_path: Optional[str] = None,
                                   title: str = 'Precision-Recall Curve'):
        """
        Trace la courbe Precision-Recall
        
        Args:
            y_true: Labels réels
            y_pred_proba: Probabilités de prédiction
            save_path: Chemin pour sauvegarder la figure
            title: Titre de la figure
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
        avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'AP = {avg_precision:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      save_path: Optional[str] = None) -> str:
        """
        Génère un rapport de classification détaillé
        
        Args:
            y_true: Labels réels
            y_pred: Prédictions
            save_path: Chemin pour sauvegarder le rapport
            
        Returns:
            Rapport de classification sous forme de string
        """
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=False
        )
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    def plot_training_history(self, history: Dict, save_path: Optional[str] = None):
        """
        Trace l'historique d'entraînement
        
        Args:
            history: Historique d'entraînement du modèle
            save_path: Chemin pour sauvegarder la figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(history['accuracy'], label='Train Accuracy')
        axes[0].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(history['loss'], label='Train Loss')
        axes[1].plot(history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self, results: Dict[str, Dict], save_path: Optional[str] = None):
        """
        Compare les performances de plusieurs modèles
        
        Args:
            results: Dictionnaire {nom_modèle: {métriques}}
            save_path: Chemin pour sauvegarder la figure
        """
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [results[model].get(metric, 0) for model in models]
            axes[i].bar(models, values, color='steelblue')
            axes[i].set_title(f'{metric.capitalize()}')
            axes[i].set_ylabel('Score')
            axes[i].set_ylim([0, 1])
            axes[i].tick_params(axis='x', rotation=45)
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_segmentation(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Évalue les performances de segmentation
        
        Args:
            y_true: Masques de vérité terrain
            y_pred: Masques prédits
            
        Returns:
            Dictionnaire de métriques de segmentation
        """
        def iou_score(y_true, y_pred):
            intersection = np.logical_and(y_true, y_pred).sum()
            union = np.logical_or(y_true, y_pred).sum()
            if union == 0:
                return 1.0
            return intersection / union
        
        def dice_score(y_true, y_pred):
            intersection = np.logical_and(y_true, y_pred).sum()
            return 2.0 * intersection / (y_true.sum() + y_pred.sum())
        
        iou = iou_score(y_true, y_pred)
        dice = dice_score(y_true, y_pred)
        
        return {
            'iou': iou,
            'dice': dice,
            'accuracy': accuracy_score(y_true.flatten(), y_pred.flatten())
        }

