# Détection Précoce de la Maladie d'Alzheimer via Deep Learning

## Description

Ce projet présente une analyse complète des méthodes de Deep Learning pour le diagnostic précoce de la maladie d'Alzheimer à partir d'imagerie médicale. Il implémente et compare différentes architectures neuronales (CNN, RNN, Transfer Learning) pour atteindre des précisions élevées sur des bases de données standards d'imagerie médicale.

## Objectifs

- Analyser les méthodes de Deep Learning (classification, segmentation et extraction de caractéristiques) pour le diagnostic précoce d'Alzheimer
- Évaluer comparativement les architectures neuronales (CNN, RNN, Transfer Learning)
- Identifier les limitations actuelles (déséquilibre de classes, prétraitement)
- Atteindre des précisions élevées (96-98%) sur des bases de données standards

## Technologies Utilisées

- **Python** : Langage de programmation principal
- **Deep Learning** : 
  - CNN (Convolutional Neural Networks)
  - RNN (Recurrent Neural Networks) - LSTM, GRU
  - Transfer Learning (VGG16, ResNet50, InceptionV3, DenseNet121)
- **Computer Vision** : Traitement et analyse d'images médicales
- **Image Segmentation** : U-Net, SegNet, Attention U-Net
- **Medical Imaging** : Support pour MRI et PET scans
- **Data Preprocessing** : Normalisation, augmentation, gestion du déséquilibre de classes
- **Datasets** : Compatible avec ADNI, OASIS, Kaggle

## Structure du Projet

```
Alzheimer/
├── data/
│   ├── raw/              # Données brutes
│   └── processed/        # Données prétraitées
├── models/
│   ├── cnn_models.py     # Architectures CNN
│   ├── transfer_learning.py  # Modèles de transfer learning
│   ├── rnn_models.py     # Modèles RNN
│   ├── segmentation_models.py  # Modèles de segmentation
│   └── saved_models/     # Modèles sauvegardés
├── results/
│   ├── figures/          # Graphiques et visualisations
│   └── reports/          # Rapports et métriques
├── logs/                 # Logs TensorBoard
├── data_preprocessing.py # Module de prétraitement
├── training.py           # Module d'entraînement
├── evaluation.py         # Module d'évaluation
├── utils.py              # Utilitaires
├── main.py               # Script principal
├── requirements.txt      # Dépendances
└── README.md            # Documentation
```

## Installation

1. Cloner le repository ou télécharger les fichiers

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Créer la structure de dossiers (automatique lors de l'exécution) :
```bash
python main.py
```

## Utilisation

### Entraînement des modèles de classification

```bash
python main.py --mode classification
```

### Entraînement des modèles de segmentation

```bash
python main.py --mode segmentation
```

### Entraînement des deux types de modèles

```bash
python main.py --mode both
```

### Options disponibles

- `--mode` : Mode d'entraînement (`classification`, `segmentation`, `both`)
- `--seed` : Graine aléatoire pour la reproductibilité (défaut: 42)
- `--data-dir` : Répertoire contenant les données (défaut: `data/raw`)

## Architectures Implémentées

### Classification

1. **SimpleCNN** : Architecture CNN basique avec plusieurs couches convolutionnelles
2. **AdvancedCNN** : CNN avec blocs résiduels pour une meilleure performance
3. **MultiScaleCNN** : CNN avec extraction de caractéristiques multi-échelle
4. **VGG16** : Transfer learning avec VGG16 pré-entraîné sur ImageNet
5. **ResNet50** : Transfer learning avec ResNet50 pré-entraîné
6. **InceptionV3** : Transfer learning avec InceptionV3
7. **DenseNet121** : Transfer learning avec DenseNet121
8. **Ensemble** : Modèle combinant plusieurs architectures

### Segmentation

1. **U-Net** : Architecture classique pour la segmentation d'images médicales
2. **SegNet** : Architecture avec encodeur-décodeur symétrique
3. **Attention U-Net** : U-Net amélioré avec mécanisme d'attention

### RNN

1. **LSTM** : Réseau LSTM pour l'analyse de séquences temporelles
2. **GRU** : Réseau GRU, variante plus légère de LSTM
3. **Bidirectional LSTM** : LSTM bidirectionnel
4. **CNN-LSTM** : Modèle hybride combinant CNN et LSTM

## Prétraitement des Données

Le module `data_preprocessing.py` offre :

- Chargement de fichiers NIfTI (format standard pour MRI/PET)
- Extraction de coupes 2D depuis volumes 3D
- Normalisation d'intensité
- Filtrage gaussien pour réduction du bruit
- Égalisation d'histogramme
- Redimensionnement des images
- Augmentation de données (rotation, zoom, flip)
- Gestion du déséquilibre de classes (SMOTE, undersampling, oversampling)

## Évaluation

Le module `evaluation.py` calcule :

- Accuracy, Precision, Recall, F1-Score
- Matrice de confusion
- Courbes ROC et AUC
- Courbes Precision-Recall
- Métriques de segmentation (IoU, Dice Score)
- Comparaison de modèles multiples

## Entraînement

Le module `training.py` fournit :

- Entraînement avec callbacks (Early Stopping, Model Checkpoint, Reduce LR)
- Fine-tuning pour les modèles de transfer learning
- Gestion des poids de classes pour le déséquilibre
- Augmentation de données pendant l'entraînement
- Support TensorBoard pour la visualisation
- Sauvegarde automatique des meilleurs modèles

## Résultats

Les résultats sont sauvegardés dans :

- `models/saved_models/` : Modèles entraînés (format .h5)
- `results/reports/` : Métriques et rapports (JSON)
- `results/figures/` : Graphiques et visualisations (PNG)
- `logs/` : Logs TensorBoard pour visualisation interactive

## Visualisation avec TensorBoard

Pour visualiser les courbes d'entraînement :

```bash
tensorboard --logdir=logs/
```

Puis ouvrir http://localhost:6006 dans votre navigateur.

## Classes de Diagnostic

Le modèle classifie les images en 4 catégories :

- **CN** (Cognitively Normal) : Sujet sain
- **MCI** (Mild Cognitive Impairment) : Déficience cognitive légère
- **AD** (Alzheimer's Disease) : Maladie d'Alzheimer
- **EMCI** (Early Mild Cognitive Impairment) : Déficience cognitive légère précoce

## Limitations et Défis

1. **Déséquilibre de classes** : Géré via SMOTE, class weights, et techniques d'échantillonnage
2. **Prétraitement** : Normalisation et filtrage essentiels pour les images médicales
3. **Taille des données** : Augmentation de données pour augmenter la taille du dataset
4. **Interprétabilité** : Les modèles de deep learning nécessitent des techniques d'interprétation

## Améliorations Futures

- Implémentation de modèles d'attention avancés
- Intégration de données multi-modales (MRI + PET)
- Techniques d'explication (Grad-CAM, SHAP)
- Optimisation hyperparamètres automatisée
- Déploiement en production avec API REST

## Références

- ADNI (Alzheimer's Disease Neuroimaging Initiative)
- OASIS (Open Access Series of Imaging Studies)
- Kaggle Alzheimer's Dataset

## Auteur

Projet de détection précoce de la maladie d'Alzheimer via Deep Learning

## Licence

Ce projet est fourni à des fins éducatives et de recherche.

