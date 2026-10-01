# Classification des stades d'Alzheimer à partir d'IRM

Classifieur d'images qui range une IRM cérébrale dans l'un de quatre stades. Le script télécharge un jeu public, entraîne un ou plusieurs modèles, puis mesure la performance sur un jeu de test jamais utilisé pour choisir le meilleur epoch.

Ce dépôt est une démonstration technique. Il ne constitue pas un outil de diagnostic.

## Classes

Le jeu [Alzheimer's Multiclass Dataset (equal and augmented)](https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented) contient environ 44 000 IRM, déjà augmentées et rééquilibrées :

| Classe | Stade |
| --- | --- |
| NonDemented | Pas de démence |
| VeryMildDemented | Démence très légère |
| MildDemented | Démence légère |
| ModerateDemented | Démence modérée |

## Méthode

1. Téléchargement du jeu avec `kagglehub`.
2. Lecture des images depuis le disque, redimensionnement en 224×224 et normalisation entre 0 et 1.
3. Découpage stratifié **70 % train / 15 % validation / 15 % test**, avant tout entraînement.
4. Entraînement avec early stopping, baisse du learning rate et sauvegarde du meilleur modèle selon la validation.
5. Poids de classes calculés sur le train seulement.
6. Métriques finales (accuracy, précision, rappel, F1, AUC) calculées une seule fois sur le test.

L'expérience de référence compare `SimpleCNN`, entraîné depuis zéro, et `ResNet50` pré-entraîné sur ImageNet.

## Lancer le projet

```bash
pip install -r requirements.txt
python main.py --models SimpleCNN,ResNet50 --epochs 15
```

`python main.py` entraîne seulement `SimpleCNN`. Les autres choix sont `AdvancedCNN`, `MultiScaleCNN` et `VGG16`.

Le premier lancement télécharge environ 400 Mo. Le jeu reste dans le cache local de `kagglehub`, il n'est pas versionné.

| Option | Défaut | Rôle |
| --- | --- | --- |
| `--models` | `SimpleCNN` | Modèles séparés par des virgules, ou `all` |
| `--epochs` | `15` | Nombre maximal d'époques |
| `--patience` | `5` | Patience de l'early stopping |
| `--batch-size` | `32` | Taille de batch |
| `--image-size` | `224` | Côté de l'image |
| `--seed` | `42` | Graine du découpage |
| `--max-per-class` | tout le jeu | Limite pour un essai rapide |

Les courbes d'entraînement se lisent avec `tensorboard --logdir=logs/`.

## Résultats

Après l'entraînement, les fichiers locaux sont :

- `results/reports/all_results.json` — métriques de test
- `results/figures/` — matrice de confusion et comparaison des modèles
- `models/saved_models/` — poids du meilleur epoch

Ces fichiers ne sont pas dans git : ils se régénèrent en relançant le script.

## Limite

Le jeu ne fournit pas d'identifiant patient, et les images sont déjà augmentées. Une IRM et sa copie peuvent donc se trouver à la fois dans l'entraînement et dans le test. Les scores sont plus élevés que sur des patients réellement nouveaux.

## Structure

```
main.py                  expérience (chargement, entraînement, test)
data_loader.py           téléchargement Kaggle et découpage
training.py              entraînement et callbacks
evaluation.py            métriques et figures
models/cnn_models.py     CNN entraînés depuis zéro
models/transfer_learning.py
example_usage.py         essai court sur un petit échantillon
```

## Stack

Python, TensorFlow / Keras, scikit-learn, kagglehub.

## Licence

Projet fourni à des fins éducatives et de démonstration.
