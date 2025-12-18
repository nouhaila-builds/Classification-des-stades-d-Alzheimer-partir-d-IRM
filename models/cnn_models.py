"""
Architectures CNN pour la classification d'images médicales
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, Optional


class SimpleCNN:
    """CNN simple pour la classification binaire/multi-classe"""
    
    @staticmethod
    def build(input_shape: Tuple[int, int, int], num_classes: int = 4, 
             dropout_rate: float = 0.5) -> keras.Model:
        """
        Construit un modèle CNN simple
        
        Args:
            input_shape: Forme de l'image d'entrée (height, width, channels)
            num_classes: Nombre de classes (4: CN, MCI, AD, EMCI)
            dropout_rate: Taux de dropout
            
        Returns:
            Modèle Keras compilé
        """
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model


class AdvancedCNN:
    """CNN avancé avec blocs résiduels"""
    
    @staticmethod
    def residual_block(x, filters: int, kernel_size: int = 3):
        """Bloc résiduel pour CNN"""
        shortcut = x
        
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        return x
    
    @staticmethod
    def build(input_shape: Tuple[int, int, int], num_classes: int = 4,
             dropout_rate: float = 0.5) -> keras.Model:
        """
        Construit un modèle CNN avancé avec blocs résiduels
        
        Args:
            input_shape: Forme de l'image d'entrée
            num_classes: Nombre de classes
            dropout_rate: Taux de dropout
            
        Returns:
            Modèle Keras compilé
        """
        inputs = layers.Input(shape=input_shape)
        
        x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        
        x = AdvancedCNN.residual_block(x, 64)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = AdvancedCNN.residual_block(x, 128)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = AdvancedCNN.residual_block(x, 256)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = AdvancedCNN.residual_block(x, 512)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        return model


class MultiScaleCNN:
    """CNN avec extraction de caractéristiques multi-échelle"""
    
    @staticmethod
    def build(input_shape: Tuple[int, int, int], num_classes: int = 4,
             dropout_rate: float = 0.5) -> keras.Model:
        """
        Construit un modèle CNN multi-échelle
        
        Args:
            input_shape: Forme de l'image d'entrée
            num_classes: Nombre de classes
            dropout_rate: Taux de dropout
            
        Returns:
            Modèle Keras compilé
        """
        inputs = layers.Input(shape=input_shape)
        
        branch1 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(inputs)
        
        branch2 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(inputs)
        branch2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
        
        branch3 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(inputs)
        branch3 = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(branch3)
        
        branch4 = layers.MaxPooling2D((3, 3), strides=1, padding='same')(inputs)
        branch4 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(branch4)
        
        x = layers.Concatenate()([branch1, branch2, branch3, branch4])
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        return model


def compile_model(model: keras.Model, learning_rate: float = 0.001,
                 optimizer: str = 'adam') -> keras.Model:
    """
    Compile un modèle avec les paramètres optimaux
    
    Args:
        model: Modèle Keras
        learning_rate: Taux d'apprentissage
        optimizer: Nom de l'optimiseur ('adam', 'sgd', 'rmsprop')
        
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

