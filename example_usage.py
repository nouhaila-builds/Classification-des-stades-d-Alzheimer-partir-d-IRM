"""
Exemple d'utilisation du projet de détection d'Alzheimer
"""

import numpy as np
from data_preprocessing import MedicalImagePreprocessor
from models.cnn_models import SimpleCNN, compile_model
from models.transfer_learning import TransferLearningModel, compile_transfer_model
from training import ModelTrainer, calculate_class_weights
from evaluation import ModelEvaluator
from utils import get_class_names, set_seed
from sklearn.model_selection import train_test_split


def example_classification():
    """
    Exemple d'entraînement d'un modèle de classification
    """
    print("="*60)
    print("EXEMPLE: Classification avec SimpleCNN")
    print("="*60)
    
    set_seed(42)
    
    input_shape = (224, 224, 1)
    num_samples = 200
    
    print("\n1. Génération de données d'exemple...")
    X = np.random.rand(num_samples, *input_shape).astype(np.float32)
    y = np.random.randint(0, 4, size=num_samples)
    
    print(f"   Forme des données: {X.shape}")
    print(f"   Distribution des classes: {np.bincount(y)}")
    
    print("\n2. Prétraitement des données...")
    preprocessor = MedicalImagePreprocessor(target_size=(224, 224))
    X_processed = []
    for i in range(len(X)):
        processed = preprocessor.preprocess_single_image(X[i])
        X_processed.append(processed)
    X_processed = np.array(X_processed)
    X_processed = np.expand_dims(X_processed, axis=-1)
    
    print("\n3. Division train/validation/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"   Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
    
    print("\n4. Construction du modèle...")
    model = compile_model(SimpleCNN.build(input_shape, num_classes=4))
    
    print("\n5. Entraînement...")
    class_weights = calculate_class_weights(y_train)
    trainer = ModelTrainer(model, 'example_simple_cnn')
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        batch_size=16,
        epochs=10,
        class_weights=class_weights
    )
    
    print("\n6. Évaluation...")
    y_pred = model.predict(X_test).argmax(axis=1)
    y_pred_proba = model.predict(X_test)
    
    evaluator = ModelEvaluator(get_class_names())
    metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
    
    print("\nRésultats:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n7. Visualisation...")
    evaluator.plot_confusion_matrix(y_test, y_pred, 
                                   save_path='results/figures/example_confusion_matrix.png')
    
    print("\nExemple terminé avec succès!")


def example_transfer_learning():
    """
    Exemple d'utilisation du transfer learning avec VGG16
    """
    print("="*60)
    print("EXEMPLE: Transfer Learning avec VGG16")
    print("="*60)
    
    set_seed(42)
    
    input_shape = (224, 224, 3)
    num_samples = 200
    
    print("\n1. Génération de données d'exemple...")
    X = np.random.rand(num_samples, *input_shape).astype(np.float32)
    y = np.random.randint(0, 4, size=num_samples)
    
    print("\n2. Division train/validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n3. Construction du modèle VGG16...")
    model = compile_transfer_model(
        TransferLearningModel.build_vgg16(input_shape, num_classes=4, freeze_base=True)
    )
    
    print("\n4. Entraînement...")
    trainer = ModelTrainer(model, 'example_vgg16')
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        batch_size=16,
        epochs=5
    )
    
    print("\n5. Évaluation...")
    y_pred = model.predict(X_val).argmax(axis=1)
    evaluator = ModelEvaluator(get_class_names())
    metrics = evaluator.calculate_metrics(y_val, y_pred)
    
    print("\nRésultats:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nExemple terminé avec succès!")


if __name__ == '__main__':
    print("Choisissez un exemple à exécuter:")
    print("1. Classification avec SimpleCNN")
    print("2. Transfer Learning avec VGG16")
    
    choice = input("\nVotre choix (1 ou 2): ")
    
    if choice == '1':
        example_classification()
    elif choice == '2':
        example_transfer_learning()
    else:
        print("Choix invalide. Exécution de l'exemple 1 par défaut.")
        example_classification()

