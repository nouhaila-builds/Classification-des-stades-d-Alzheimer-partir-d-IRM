"""
Modèles de Transfer Learning pour la détection d'Alzheimer
Utilise VGG16, ResNet50 et autres architectures pré-entraînées
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, DenseNet121
from typing import Tuple, Optional


class TransferLearningModel:
    """Classe de base pour les modèles de transfer learning"""
    
    @staticmethod
    def build_vgg16(input_shape: Tuple[int, int, int], num_classes: int = 4,
                   freeze_base: bool = True, dropout_rate: float = 0.5) -> keras.Model:
        """
        Construit un modèle basé sur VGG16 pré-entraîné
        
        Args:
            input_shape: Forme de l'image d'entrée
            num_classes: Nombre de classes
            freeze_base: Si True, gèle les poids de la base
            dropout_rate: Taux de dropout
            
        Returns:
            Modèle Keras compilé
        """
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        if freeze_base:
            base_model.trainable = False
        else:
            for layer in base_model.layers[:-4]:
                layer.trainable = False
        
        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    @staticmethod
    def build_resnet50(input_shape: Tuple[int, int, int], num_classes: int = 4,
                      freeze_base: bool = True, dropout_rate: float = 0.5) -> keras.Model:
        """
        Construit un modèle basé sur ResNet50 pré-entraîné
        
        Args:
            input_shape: Forme de l'image d'entrée
            num_classes: Nombre de classes
            freeze_base: Si True, gèle les poids de la base
            dropout_rate: Taux de dropout
            
        Returns:
            Modèle Keras compilé
        """
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        if freeze_base:
            base_model.trainable = False
        else:
            for layer in base_model.layers[:-10]:
                layer.trainable = False
        
        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    @staticmethod
    def build_inceptionv3(input_shape: Tuple[int, int, int], num_classes: int = 4,
                         freeze_base: bool = True, dropout_rate: float = 0.5) -> keras.Model:
        """
        Construit un modèle basé sur InceptionV3 pré-entraîné
        
        Args:
            input_shape: Forme de l'image d'entrée
            num_classes: Nombre de classes
            freeze_base: Si True, gèle les poids de la base
            dropout_rate: Taux de dropout
            
        Returns:
            Modèle Keras compilé
        """
        base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        if freeze_base:
            base_model.trainable = False
        else:
            for layer in base_model.layers[:-20]:
                layer.trainable = False
        
        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    @staticmethod
    def build_densenet121(input_shape: Tuple[int, int, int], num_classes: int = 4,
                         freeze_base: bool = True, dropout_rate: float = 0.5) -> keras.Model:
        """
        Construit un modèle basé sur DenseNet121 pré-entraîné
        
        Args:
            input_shape: Forme de l'image d'entrée
            num_classes: Nombre de classes
            freeze_base: Si True, gèle les poids de la base
            dropout_rate: Taux de dropout
            
        Returns:
            Modèle Keras compilé
        """
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        if freeze_base:
            base_model.trainable = False
        else:
            for layer in base_model.layers[:-10]:
                layer.trainable = False
        
        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        return model


class EnsembleTransferLearning:
    """Modèle d'ensemble combinant plusieurs architectures de transfer learning"""
    
    @staticmethod
    def build_ensemble(input_shape: Tuple[int, int, int], num_classes: int = 4,
                      dropout_rate: float = 0.5) -> keras.Model:
        """
        Construit un modèle d'ensemble combinant VGG16, ResNet50 et DenseNet121
        
        Args:
            input_shape: Forme de l'image d'entrée
            num_classes: Nombre de classes
            dropout_rate: Taux de dropout
            
        Returns:
            Modèle Keras compilé
        """
        inputs = keras.Input(shape=input_shape)
        
        vgg16_base = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
        vgg16_base.trainable = False
        vgg16_out = vgg16_base(inputs, training=False)
        vgg16_out = layers.GlobalAveragePooling2D()(vgg16_out)
        
        resnet50_base = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
        resnet50_base.trainable = False
        resnet50_out = resnet50_base(inputs, training=False)
        resnet50_out = layers.GlobalAveragePooling2D()(resnet50_out)
        
        densenet121_base = DenseNet121(weights='imagenet', include_top=False, input_tensor=inputs)
        densenet121_base.trainable = False
        densenet121_out = densenet121_base(inputs, training=False)
        densenet121_out = layers.GlobalAveragePooling2D()(densenet121_out)
        
        combined = layers.Concatenate()([vgg16_out, resnet50_out, densenet121_out])
        combined = layers.Dense(512, activation='relu')(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(dropout_rate)(combined)
        combined = layers.Dense(256, activation='relu')(combined)
        combined = layers.Dropout(dropout_rate)(combined)
        outputs = layers.Dense(num_classes, activation='softmax')(combined)
        
        model = keras.Model(inputs, outputs)
        return model


def compile_transfer_model(model: keras.Model, learning_rate: float = 0.0001,
                          optimizer: str = 'adam') -> keras.Model:
    """
    Compile un modèle de transfer learning avec les paramètres optimaux
    
    Args:
        model: Modèle Keras
        learning_rate: Taux d'apprentissage (plus faible pour transfer learning)
        optimizer: Nom de l'optimiseur
        
    Returns:
        Modèle compilé
    """
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    
    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
    )
    
    return model

