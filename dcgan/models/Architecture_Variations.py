#!/usr/bin/env python3
"""
Train the DCGAN with the modified architecture, modified the DCGAN architecture(number of layers,filter size and strides)
"""

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import time
import wandb
import os

# Initialize WandB for experiment tracking
wandb.login()
wandb.init(project='DCGAN', name='Architecture Variations')

# Configuration settings
config = wandb.config
config.BUFFER_SIZE = 60000
config.BATCH_SIZE = 256
config.EPOCHS = 10
config.noise_dim = 100
config.num_to_generate = 16

# Load MNIST dataset
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

# Preprocess and normalize images
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

# Create TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE)

# Generator model architecture
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(config.noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model

# Discriminator model architecture
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(28, 28, 1)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# Instantiate generator and discriminator models
generator = make_generator_model()
discriminator = make_discriminator_model()

# Binary cross-entropy loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Discriminator loss function
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Generator loss function
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Adam optimizers for generator and discriminator
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Random seed for image generation
seed = tf.random.normal([config.num_to_generate, config.noise_dim])

# Training step function
@tf.function
def train_step(images):
    noise = tf.random.normal([config.BATCH_SIZE, config.noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Training function
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        gen_losses, disc_losses = [], []

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)

        avg_gen_loss = sum(gen_losses) / len(gen_losses)
        avg_disc_loss = sum(disc_losses) / len(disc_losses)

        print('Epoch {}: Generator Loss: {}, Discriminator Loss: {}, Time: {}'.format(epoch + 1, avg_gen_loss, avg_disc_loss, time.time() - start))

        generate_and_save_images(generator, epoch + 1, seed)

        wandb.log({"Generator Loss": avg_gen_loss, "Discriminator Loss": avg_disc_loss, "Epoch": epoch + 1, "Time": time.time() - start})

# Image generation and saving function
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    img_path = f'image_at_epoch_{epoch:04d}.png'
    plt.savefig(img_path)
    plt.show()

    wandb.log({"Generated Images": [wandb.Image(img_path, caption=f'Epoch {epoch}')]})


# Start training with modified architecture
train(train_dataset, config.EPOCHS)

# Finish WandB run
wandb.finish()

