"""
Détection des stades d'Alzheimer à partir d'IRM.

Le script télécharge le jeu Kaggle
aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented,
puis entraîne et compare des modèles de classification.
Les métriques finales sont calculées sur un jeu de test jamais utilisé
pour l'entraînement ni pour le choix du meilleur epoch.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")

import numpy as np

from data_loader import (
    CLASS_NAMES,
    collect_samples,
    download_dataset,
    find_image_root,
    make_dataset,
    split_samples,
)
from evaluation import ModelEvaluator
from models.cnn_models import AdvancedCNN, MultiScaleCNN, SimpleCNN, compile_model
from models.transfer_learning import TransferLearningModel, compile_transfer_model
from training import ModelTrainer, calculate_class_weights
from utils import create_directories, get_class_names, print_model_summary, save_results, set_seed

MODEL_CHOICES = ["SimpleCNN", "AdvancedCNN", "MultiScaleCNN", "VGG16", "ResNet50"]


def build_model(name: str, input_shape, num_classes):
    """Construit et compile l'architecture demandée."""
    if name == "SimpleCNN":
        return compile_model(SimpleCNN.build(input_shape, num_classes))
    if name == "AdvancedCNN":
        return compile_model(AdvancedCNN.build(input_shape, num_classes))
    if name == "MultiScaleCNN":
        return compile_model(MultiScaleCNN.build(input_shape, num_classes))
    if name == "VGG16":
        return compile_transfer_model(
            TransferLearningModel.build_vgg16(input_shape, num_classes)
        )
    if name == "ResNet50":
        return compile_transfer_model(
            TransferLearningModel.build_resnet50(input_shape, num_classes)
        )
    raise ValueError(f"Modèle inconnu: {name}. Choix: {', '.join(MODEL_CHOICES)}.")


def evaluate_on_test(model, test_ds, class_names, model_name):
    """Évalue le modèle restauré (meilleur epoch) sur le jeu de test."""
    y_true = []
    y_prob_parts = []
    for images, labels in test_ds:
        y_true.append(labels.numpy())
        y_prob_parts.append(model.predict(images, verbose=0))

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob_parts)
    y_pred = y_prob.argmax(axis=1)

    evaluator = ModelEvaluator(class_names)
    metrics = evaluator.calculate_metrics(y_true, y_pred, y_prob)
    evaluator.plot_confusion_matrix(
        y_true,
        y_pred,
        save_path=f"results/figures/{model_name}_confusion_matrix.png",
        title=f"Matrice de confusion — {model_name}",
    )
    return {key: float(value) for key, value in metrics.items()}


def train_classification_models(
    train_ds,
    val_ds,
    test_ds,
    y_train,
    input_shape,
    model_names,
    epochs,
    patience,
):
    """Entraîne les modèles choisis et les évalue sur le test."""
    results = {}
    class_names = get_class_names()
    num_classes = len(class_names)
    class_weights = calculate_class_weights(y_train)
    print(f"Poids de classes (train uniquement): {class_weights}")

    for model_name in model_names:
        print(f"\n{'=' * 60}")
        print(f"Entraînement de {model_name}")
        print(f"{'=' * 60}")

        model = build_model(model_name, input_shape, num_classes)
        print_model_summary(model)

        trainer = ModelTrainer(model, model_name)
        history = trainer.train(
            train_ds,
            X_val=val_ds,
            epochs=epochs,
            class_weights=class_weights,
            patience=patience,
        )
        trainer.save_model()

        metrics = evaluate_on_test(model, test_ds, class_names, model_name)
        results[model_name] = {
            "history": {key: [float(value) for value in values] for key, values in history.items()},
            "metrics": metrics,
        }

        print(f"\nMétriques de test pour {model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    return results


def compare_all_models(results):
    """Compare les métriques de test des modèles entraînés."""
    print("\n" + "=" * 60)
    print("COMPARAISON DES MODÈLES (JEU DE TEST)")
    print("=" * 60)

    metrics_dict = {
        model_name: result["metrics"]
        for model_name, result in results.items()
        if "metrics" in result
    }

    evaluator = ModelEvaluator(get_class_names())
    evaluator.compare_models(metrics_dict, save_path="results/figures/model_comparison.png")

    print(f"\n{'Modèle':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 68)
    for model_name, metrics in metrics_dict.items():
        print(
            f"{model_name:<20} "
            f"{metrics.get('accuracy', 0):<12.4f} "
            f"{metrics.get('precision', 0):<12.4f} "
            f"{metrics.get('recall', 0):<12.4f} "
            f"{metrics.get('f1_score', 0):<12.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Classification des stades d'Alzheimer sur IRM (jeu Kaggle augmenté)"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="SimpleCNN",
        help=(
            "Modèles séparés par des virgules. "
            f"Choix: {', '.join(MODEL_CHOICES)}, ou all. Défaut: SimpleCNN."
        ),
    )
    parser.add_argument("--epochs", type=int, default=15, help="Nombre maximal d'époques")
    parser.add_argument("--patience", type=int, default=5, help="Patience de l'early stopping")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille de batch")
    parser.add_argument("--image-size", type=int, default=224, help="Taille des images après redimensionnement")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Limite d'images par classe, réservée aux essais rapides",
    )
    args = parser.parse_args()

    if args.models.strip().lower() == "all":
        model_names = list(MODEL_CHOICES)
    else:
        model_names = [name.strip() for name in args.models.split(",") if name.strip()]
        unknown = [name for name in model_names if name not in MODEL_CHOICES]
        if unknown:
            parser.error(f"Modèles inconnus: {', '.join(unknown)}. Choix: {', '.join(MODEL_CHOICES)}.")

    set_seed(args.seed)
    create_directories()

    print("=" * 60)
    print("CLASSIFICATION DES STADES D'ALZHEIMER")
    print("Jeu: aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented")
    print("Classes:", ", ".join(CLASS_NAMES))
    print("=" * 60)
    print(
        "\nCe jeu est une version déjà augmentée et rééquilibrée. "
        "Un découpage aléatoire peut placer une image et son augmentation "
        "des deux côtés du split. Les scores sont donc optimistes par rapport "
        "à un test sur des patients jamais vus."
    )

    dataset_path = download_dataset()
    image_root = find_image_root(dataset_path)
    print(f"Dossier des images: {image_root}")

    print("\nInventaire des images...")
    paths, labels = collect_samples(
        image_root,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )

    print("\nDécoupage stratifié (70 % train, 15 % validation, 15 % test)...")
    splits = split_samples(paths, labels, seed=args.seed, test_size=0.15, val_size=0.15)

    input_shape = (args.image_size, args.image_size, 3)
    train_ds = make_dataset(
        *splits["train"],
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_ds = make_dataset(
        *splits["val"],
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_ds = make_dataset(
        *splits["test"],
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
    )

    results = train_classification_models(
        train_ds,
        val_ds,
        test_ds,
        splits["train"][1],
        input_shape,
        model_names,
        epochs=args.epochs,
        patience=args.patience,
    )

    compare_all_models(results)
    os.makedirs("results/reports", exist_ok=True)
    save_results(results, "results/reports/all_results.json")

    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT TERMINÉ")
    print("=" * 60)
    print("Résultats:")
    print("  - models/saved_models/ (modèles)")
    print("  - results/reports/all_results.json (métriques de test)")
    print("  - results/figures/ (matrices de confusion et comparaison)")
    print("  - logs/ (TensorBoard)")
    print("\nVisualisation: tensorboard --logdir=logs/")


if __name__ == "__main__":
    main()
