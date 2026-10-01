"""
Modèles de segmentation pour l'analyse d'images médicales
Implémentation de U-Net et SegNet
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, Optional
import tensorflow.keras.backend as K


class UNet:
    """Architecture U-Net pour la segmentation d'images médicales"""
    
    @staticmethod
    def conv_block(x, filters: int, kernel_size: int = 3, padding: str = 'same'):
        """Bloc de convolution avec activation et batch normalization"""
        x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x
    
    @staticmethod
    def build(input_shape: Tuple[int, int, int], num_classes: int = 1,
             filters: int = 64) -> keras.Model:
        """
        Construit un modèle U-Net
        
        Args:
            input_shape: Forme de l'image d'entrée
            num_classes: Nombre de classes de segmentation
            filters: Nombre de filtres de base
            
        Returns:
            Modèle Keras compilé
        """
        inputs = layers.Input(shape=input_shape)
        
        c1 = UNet.conv_block(inputs, filters)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = UNet.conv_block(p1, filters * 2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = UNet.conv_block(p2, filters * 4)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        
        c4 = UNet.conv_block(p3, filters * 8)
        p4 = layers.MaxPooling2D((2, 2))(c4)
        
        c5 = UNet.conv_block(p4, filters * 16)
        
        u6 = layers.Conv2DTranspose(filters * 8, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.Concatenate()([u6, c4])
        c6 = UNet.conv_block(u6, filters * 8)
        
        u7 = layers.Conv2DTranspose(filters * 4, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.Concatenate()([u7, c3])
        c7 = UNet.conv_block(u7, filters * 4)
        
        u8 = layers.Conv2DTranspose(filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.Concatenate()([u8, c2])
        c8 = UNet.conv_block(u8, filters * 2)
        
        u9 = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.Concatenate()([u9, c1])
        c9 = UNet.conv_block(u9, filters)
        
        if num_classes == 1:
            outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
        else:
            outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)
        
        model = models.Model(inputs, outputs)
        return model


class SegNet:
    """Architecture SegNet pour la segmentation d'images médicales"""
    
    @staticmethod
    def encoder_block(x, filters: int, pool_size: Tuple[int, int] = (2, 2)):
        """Bloc encodeur de SegNet"""
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size)(x)
        return x
    
    @staticmethod
    def decoder_block(x, filters: int, upsample_size: Tuple[int, int] = (2, 2)):
        """Bloc décodeur de SegNet"""
        x = layers.UpSampling2D(upsample_size)(x)
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x
    
    @staticmethod
    def build(input_shape: Tuple[int, int, int], num_classes: int = 1,
             filters: int = 64) -> keras.Model:
        """
        Construit un modèle SegNet
        
        Args:
            input_shape: Forme de l'image d'entrée
            num_classes: Nombre de classes de segmentation
            filters: Nombre de filtres de base
            
        Returns:
            Modèle Keras compilé
        """
        inputs = layers.Input(shape=input_shape)
        
        e1 = SegNet.encoder_block(inputs, filters)
        e2 = SegNet.encoder_block(e1, filters * 2)
        e3 = SegNet.encoder_block(e2, filters * 4)
        e4 = SegNet.encoder_block(e3, filters * 8)
        e5 = SegNet.encoder_block(e4, filters * 16)
        
        d5 = SegNet.decoder_block(e5, filters * 16)
        d4 = SegNet.decoder_block(d5, filters * 8)
        d3 = SegNet.decoder_block(d4, filters * 4)
        d2 = SegNet.decoder_block(d3, filters * 2)
        d1 = SegNet.decoder_block(d2, filters)
        
        if num_classes == 1:
            outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d1)
        else:
            outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(d1)
        
        model = models.Model(inputs, outputs)
        return model


class AttentionUNet:
    """U-Net avec mécanisme d'attention pour améliorer la segmentation"""
    
    @staticmethod
    def attention_block(x, g, filters: int):
        """Bloc d'attention pour U-Net"""
        theta_x = layers.Conv2D(filters, (1, 1), padding='same')(x)
        phi_g = layers.Conv2D(filters, (1, 1), padding='same')(g)
        
        f = layers.Activation('relu')(layers.Add()([theta_x, phi_g]))
        psi_f = layers.Conv2D(1, (1, 1), padding='same')(f)
        rate = layers.Activation('sigmoid')(psi_f)
        
        att_x = layers.Multiply()([x, rate])
        return att_x
    
    @staticmethod
    def build(input_shape: Tuple[int, int, int], num_classes: int = 1,
             filters: int = 64) -> keras.Model:
        """
        Construit un modèle Attention U-Net
        
        Args:
            input_shape: Forme de l'image d'entrée
            num_classes: Nombre de classes de segmentation
            filters: Nombre de filtres de base
            
        Returns:
            Modèle Keras compilé
        """
        inputs = layers.Input(shape=input_shape)
        
        c1 = UNet.conv_block(inputs, filters)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = UNet.conv_block(p1, filters * 2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = UNet.conv_block(p2, filters * 4)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        
        c4 = UNet.conv_block(p3, filters * 8)
        p4 = layers.MaxPooling2D((2, 2))(c4)
        
        c5 = UNet.conv_block(p4, filters * 16)
        
        u6 = layers.Conv2DTranspose(filters * 8, (2, 2), strides=(2, 2), padding='same')(c5)
        att6 = AttentionUNet.attention_block(c4, u6, filters * 8)
        u6 = layers.Concatenate()([u6, att6])
        c6 = UNet.conv_block(u6, filters * 8)
        
        u7 = layers.Conv2DTranspose(filters * 4, (2, 2), strides=(2, 2), padding='same')(c6)
        att7 = AttentionUNet.attention_block(c3, u7, filters * 4)
        u7 = layers.Concatenate()([u7, att7])
        c7 = UNet.conv_block(u7, filters * 4)
        
        u8 = layers.Conv2DTranspose(filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
        att8 = AttentionUNet.attention_block(c2, u8, filters * 2)
        u8 = layers.Concatenate()([u8, att8])
        c8 = UNet.conv_block(u8, filters * 2)
        
        u9 = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c8)
        att9 = AttentionUNet.attention_block(c1, u9, filters)
        u9 = layers.Concatenate()([u9, att9])
        c9 = UNet.conv_block(u9, filters)
        
        if num_classes == 1:
            outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
        else:
            outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)
        
        model = models.Model(inputs, outputs)
        return model


def iou_metric(y_true, y_pred):
    """Métrique IoU personnalisée pour la segmentation"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    
    iou = intersection / (union + K.epsilon())
    return iou


def compile_segmentation_model(model: keras.Model, num_classes: int = 1,
                              learning_rate: float = 0.001) -> keras.Model:
    """
    Compile un modèle de segmentation
    
    Args:
        model: Modèle Keras
        num_classes: Nombre de classes
        learning_rate: Taux d'apprentissage
        
    Returns:
        Modèle compilé
    """
    if num_classes == 1:
        loss = 'binary_crossentropy'
        metrics = ['accuracy', iou_metric]
    else:
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy', 'sparse_categorical_crossentropy']
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )
    
    return model

