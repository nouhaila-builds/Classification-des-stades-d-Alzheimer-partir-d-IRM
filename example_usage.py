"""
Exemple court sur un sous-ensemble du jeu Kaggle.

Pour l'entraînement complet, utiliser main.py.
--max-per-class ici ne sert qu'à vérifier que le chargement fonctionne.
"""

import matplotlib
matplotlib.use("Agg")

from data_loader import (
    collect_samples,
    download_dataset,
    find_image_root,
    make_dataset,
    split_samples,
)
from evaluation import ModelEvaluator
from models.cnn_models import SimpleCNN, compile_model
from training import ModelTrainer, calculate_class_weights
from utils import get_class_names, set_seed


def example_classification(max_per_class: int = 32, epochs: int = 2):
    """
    Entraîne SimpleCNN sur un petit échantillon réel du jeu Kaggle.
    """
    print("=" * 60)
    print("EXEMPLE: SimpleCNN sur un extrait du jeu Kaggle")
    print("=" * 60)

    set_seed(42)
    image_size = 128
    input_shape = (image_size, image_size, 3)

    print("\n1. Téléchargement du jeu...")
    image_root = find_image_root(download_dataset())

    print("\n2. Échantillon et découpage...")
    paths, labels = collect_samples(image_root, max_per_class=max_per_class, seed=42)
    splits = split_samples(paths, labels, seed=42)

    train_ds = make_dataset(*splits["train"], image_size=image_size, batch_size=16, shuffle=True)
    val_ds = make_dataset(*splits["val"], image_size=image_size, batch_size=16)
    test_ds = make_dataset(*splits["test"], image_size=image_size, batch_size=16)

    print("\n3. Construction et entraînement de SimpleCNN...")
    model = compile_model(SimpleCNN.build(input_shape, num_classes=len(get_class_names())))
    class_weights = calculate_class_weights(splits["train"][1])
    trainer = ModelTrainer(model, "example_simple_cnn")
    trainer.train(
        train_ds,
        X_val=val_ds,
        epochs=epochs,
        class_weights=class_weights,
        patience=2,
    )

    print("\n4. Évaluation sur le test...")
    y_true = []
    y_pred = []
    for images, batch_labels in test_ds:
        probabilities = model.predict(images, verbose=0)
        y_true.extend(batch_labels.numpy().tolist())
        y_pred.extend(probabilities.argmax(axis=1).tolist())

    evaluator = ModelEvaluator(get_class_names())
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    evaluator.plot_confusion_matrix(
        y_true,
        y_pred,
        save_path="results/figures/example_confusion_matrix.png",
    )
    print("\nExemple terminé. Lancer python main.py pour le jeu complet.")


if __name__ == "__main__":
    example_classification()

