"""
Module de prétraitement des données d'imagerie médicale
pour la détection précoce de la maladie d'Alzheimer
"""

import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
import nibabel as nib
from typing import Tuple, List, Optional
import os


class MedicalImagePreprocessor:
    """Classe pour le prétraitement des images médicales (MRI, PET)"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224), normalize: bool = True):
        """
        Initialise le préprocesseur
        
        Args:
            target_size: Taille cible des images après redimensionnement
            normalize: Si True, normalise les images
        """
        self.target_size = target_size
        self.normalize = normalize
        self.scaler = StandardScaler()
    
    def load_nifti(self, file_path: str) -> np.ndarray:
        """
        Charge un fichier NIfTI (format standard pour MRI/PET)
        
        Args:
            file_path: Chemin vers le fichier .nii ou .nii.gz
            
        Returns:
            Array numpy contenant les données d'image
        """
        try:
            nii_img = nib.load(file_path)
            data = nii_img.get_fdata()
            return data
        except Exception as e:
            print(f"Erreur lors du chargement de {file_path}: {e}")
            return None
    
    def extract_slice(self, volume: np.ndarray, slice_idx: Optional[int] = None, 
                     axis: int = 2) -> np.ndarray:
        """
        Extrait une coupe 2D d'un volume 3D
        
        Args:
            volume: Volume 3D
            slice_idx: Index de la coupe (si None, prend la coupe centrale)
            axis: Axe le long duquel extraire la coupe
            
        Returns:
            Image 2D
        """
        if slice_idx is None:
            slice_idx = volume.shape[axis] // 2
        
        if axis == 0:
            slice_img = volume[slice_idx, :, :]
        elif axis == 1:
            slice_img = volume[:, slice_idx, :]
        else:
            slice_img = volume[:, :, slice_idx]
        
        return slice_img
    
    def normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """
        Normalise l'intensité de l'image entre 0 et 1
        
        Args:
            image: Image à normaliser
            
        Returns:
            Image normalisée
        """
        img_min = np.min(image)
        img_max = np.max(image)
        
        if img_max - img_min > 0:
            normalized = (image - img_min) / (img_max - img_min)
        else:
            normalized = image
        
        return normalized
    
    def apply_gaussian_filter(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Applique un filtre gaussien pour réduire le bruit
        
        Args:
            image: Image d'entrée
            sigma: Écart-type du filtre gaussien
            
        Returns:
            Image filtrée
        """
        return ndimage.gaussian_filter(image, sigma=sigma)
    
    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Applique l'égalisation d'histogramme pour améliorer le contraste
        
        Args:
            image: Image d'entrée (doit être uint8)
            
        Returns:
            Image avec histogramme égalisé
        """
        if image.dtype != np.uint8:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        equalized = cv2.equalizeHist(image_uint8)
        return equalized.astype(np.float32) / 255.0
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Redimensionne l'image à la taille cible
        
        Args:
            image: Image à redimensionner
            
        Returns:
            Image redimensionnée
        """
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_CUBIC)
        return resized
    
    def augment_data(self, image: np.ndarray, rotation_range: int = 15, 
                    zoom_range: float = 0.1, flip: bool = True) -> List[np.ndarray]:
        """
        Génère des augmentations de données pour augmenter la taille du dataset
        
        Args:
            image: Image d'entrée
            rotation_range: Plage de rotation en degrés
            zoom_range: Plage de zoom
            flip: Si True, ajoute des images retournées
            
        Returns:
            Liste d'images augmentées
        """
        augmented = [image]
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        if rotation_range > 0:
            angle = np.random.uniform(-rotation_range, rotation_range)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))
            augmented.append(rotated)
        
        if zoom_range > 0:
            zoom_factor = np.random.uniform(1 - zoom_range, 1 + zoom_range)
            M = cv2.getRotationMatrix2D(center, 0, zoom_factor)
            zoomed = cv2.warpAffine(image, M, (w, h))
            augmented.append(zoomed)
        
        if flip:
            flipped_h = cv2.flip(image, 1)
            flipped_v = cv2.flip(image, 0)
            augmented.extend([flipped_h, flipped_v])
        
        return augmented
    
    def preprocess_single_image(self, image: np.ndarray, apply_augmentation: bool = False) -> np.ndarray:
        """
        Prétraite une seule image avec toutes les étapes
        
        Args:
            image: Image d'entrée
            apply_augmentation: Si True, applique l'augmentation
            
        Returns:
            Image prétraitée
        """
        if len(image.shape) == 3:
            image = self.extract_slice(image)
        
        if self.normalize:
            image = self.normalize_intensity(image)
        
        image = self.apply_gaussian_filter(image)
        image = self.resize_image(image)
        
        if apply_augmentation:
            augmented = self.augment_data(image)
            return augmented[0]
        
        return image
    
    def preprocess_batch(self, images: List[np.ndarray], labels: Optional[List[int]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prétraite un batch d'images
        
        Args:
            images: Liste d'images
            labels: Liste de labels (optionnel)
            
        Returns:
            Tuple (images prétraitées, labels)
        """
        processed_images = []
        processed_labels = []
        
        for i, img in enumerate(images):
            processed = self.preprocess_single_image(img)
            processed_images.append(processed)
            
            if labels is not None:
                processed_labels.append(labels[i])
        
        processed_images = np.array(processed_images)
        
        if len(processed_images.shape) == 3:
            processed_images = np.expand_dims(processed_images, axis=-1)
        
        if labels is not None:
            return processed_images, np.array(processed_labels)
        
        return processed_images, None
    
    def handle_class_imbalance(self, X: np.ndarray, y: np.ndarray, 
                              strategy: str = 'smote') -> Tuple[np.ndarray, np.ndarray]:
        """
        Gère le déséquilibre de classes dans le dataset
        
        Args:
            X: Features
            y: Labels
            strategy: Stratégie ('smote', 'undersample', 'oversample')
            
        Returns:
            Tuple (X_balanced, y_balanced)
        """
        from collections import Counter
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.over_sampling import RandomOverSampler
        
        class_counts = Counter(y)
        print(f"Distribution des classes avant équilibrage: {class_counts}")
        
        if strategy == 'smote':
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(
                X.reshape(X.shape[0], -1), y
            )
            X_balanced = X_balanced.reshape(-1, *X.shape[1:])
        elif strategy == 'undersample':
            undersampler = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = undersampler.fit_resample(
                X.reshape(X.shape[0], -1), y
            )
            X_balanced = X_balanced.reshape(-1, *X.shape[1:])
        elif strategy == 'oversample':
            oversampler = RandomOverSampler(random_state=42)
            X_balanced, y_balanced = oversampler.fit_resample(
                X.reshape(X.shape[0], -1), y
            )
            X_balanced = X_balanced.reshape(-1, *X.shape[1:])
        else:
            return X, y
        
        class_counts_after = Counter(y_balanced)
        print(f"Distribution des classes après équilibrage: {class_counts_after}")
        
        return X_balanced, y_balanced

