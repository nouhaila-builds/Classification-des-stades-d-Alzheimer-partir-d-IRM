"""
Modèles RNN pour l'analyse temporelle de séquences d'images médicales
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, Optional


class LSTMClassifier:
    """Classificateur LSTM pour séquences d'images"""
    
    @staticmethod
    def build(input_shape: Tuple[int, int], num_classes: int = 4,
             lstm_units: int = 128, dropout_rate: float = 0.5) -> keras.Model:
        """
        Construit un modèle LSTM pour la classification
        
        Args:
            input_shape: Forme de la séquence (timesteps, features)
            num_classes: Nombre de classes
            lstm_units: Nombre d'unités LSTM
            dropout_rate: Taux de dropout
            
        Returns:
            Modèle Keras compilé
        """
        model = models.Sequential([
            layers.LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
            layers.Dropout(dropout_rate),
            layers.LSTM(lstm_units, return_sequences=True),
            layers.Dropout(dropout_rate),
            layers.LSTM(lstm_units // 2),
            layers.Dropout(dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(128, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model


class GRUClassifier:
    """Classificateur GRU pour séquences d'images"""
    
    @staticmethod
    def build(input_shape: Tuple[int, int], num_classes: int = 4,
             gru_units: int = 128, dropout_rate: float = 0.5) -> keras.Model:
        """
        Construit un modèle GRU pour la classification
        
        Args:
            input_shape: Forme de la séquence (timesteps, features)
            num_classes: Nombre de classes
            gru_units: Nombre d'unités GRU
            dropout_rate: Taux de dropout
            
        Returns:
            Modèle Keras compilé
        """
        model = models.Sequential([
            layers.GRU(gru_units, return_sequences=True, input_shape=input_shape),
            layers.Dropout(dropout_rate),
            layers.GRU(gru_units, return_sequences=True),
            layers.Dropout(dropout_rate),
            layers.GRU(gru_units // 2),
            layers.Dropout(dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(128, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model


class BidirectionalLSTM:
    """Classificateur LSTM bidirectionnel"""
    
    @staticmethod
    def build(input_shape: Tuple[int, int], num_classes: int = 4,
             lstm_units: int = 128, dropout_rate: float = 0.5) -> keras.Model:
        """
        Construit un modèle LSTM bidirectionnel
        
        Args:
            input_shape: Forme de la séquence
            num_classes: Nombre de classes
            lstm_units: Nombre d'unités LSTM
            dropout_rate: Taux de dropout
            
        Returns:
            Modèle Keras compilé
        """
        model = models.Sequential([
            layers.Bidirectional(
                layers.LSTM(lstm_units, return_sequences=True),
                input_shape=input_shape
            ),
            layers.Dropout(dropout_rate),
            layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True)),
            layers.Dropout(dropout_rate),
            layers.Bidirectional(layers.LSTM(lstm_units // 2)),
            layers.Dropout(dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(128, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model


class CNNLSTM:
    """Modèle hybride CNN-LSTM pour l'analyse de séquences d'images"""
    
    @staticmethod
    def build_cnn_feature_extractor(input_shape: Tuple[int, int, int]) -> keras.Model:
        """
        Extrait les caractéristiques d'une image avec CNN
        
        Args:
            input_shape: Forme de l'image (height, width, channels)
            
        Returns:
            Modèle CNN pour extraction de caractéristiques
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
            
            layers.GlobalAveragePooling2D()
        ])
        
        return model
    
    @staticmethod
    def build(sequence_length: int, image_shape: Tuple[int, int, int],
             num_classes: int = 4, lstm_units: int = 128,
             dropout_rate: float = 0.5) -> keras.Model:
        """
        Construit un modèle CNN-LSTM
        
        Args:
            sequence_length: Longueur de la séquence d'images
            image_shape: Forme d'une image
            num_classes: Nombre de classes
            lstm_units: Nombre d'unités LSTM
            dropout_rate: Taux de dropout
            
        Returns:
            Modèle Keras compilé
        """
        cnn_model = CNNLSTM.build_cnn_feature_extractor(image_shape)
        
        sequence_input = layers.Input(shape=(sequence_length, *image_shape))
        time_distributed = layers.TimeDistributed(cnn_model)(sequence_input)
        
        lstm_out = layers.LSTM(lstm_units, return_sequences=True)(time_distributed)
        lstm_out = layers.Dropout(dropout_rate)(lstm_out)
        lstm_out = layers.LSTM(lstm_units // 2)(lstm_out)
        lstm_out = layers.Dropout(dropout_rate)(lstm_out)
        
        dense = layers.Dense(256, activation='relu')(lstm_out)
        dense = layers.Dropout(dropout_rate)(dense)
        dense = layers.Dense(128, activation='relu')(dense)
        dense = layers.Dropout(dropout_rate)(dense)
        outputs = layers.Dense(num_classes, activation='softmax')(dense)
        
        model = models.Model(sequence_input, outputs)
        return model


def compile_rnn_model(model: keras.Model, learning_rate: float = 0.001,
                     optimizer: str = 'adam') -> keras.Model:
    """
    Compile un modèle RNN
    
    Args:
        model: Modèle Keras
        learning_rate: Taux d'apprentissage
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

