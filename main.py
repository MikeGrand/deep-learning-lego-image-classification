import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import os
import random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
import keras
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--dataset_path", required=True , help="path to dataset")
args = vars(parser.parse_args())

data_dir = (args["dataset_path"])
Name = ['4589','6558','11212','11211']

# Listas para almacenar imágenes y etiquetas
datax0 = []
datay0 = []
count = 0

# Iterar sobre cada carpeta (clase)
for file in Name:
    path = os.path.join(data_dir, file)
    all_images = os.listdir(path)

    all_images.sort(key=lambda x: int(x.split('.')[0]))


    # Usar solo la primera mitad de las imágenes
    half_images = all_images[:len(all_images)]

    # Procesar cada imagen seleccionada
    for im in half_images:
        image = load_img(os.path.join(path, im), color_mode='rgb', target_size=(64, 64))
        image = img_to_array(image)
        # Convertir a uint8 para procesar con OpenCV
        image = (image * 255).astype(np.uint8)
        # === DETECCIÓN DE BORDES ===
        # 3. Convertir a escala de grises
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 4. Aplicar desenfoque para reducir ruido
        blur = cv2.GaussianBlur(gris, (5, 5), 0)

        # 5. Aplicar umbral para detectar figura
        _, umbral = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 6. Encontrar contornos
        contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 7. Dibujar el contorno más grande (en verde)
        if contornos:
            contorno_principal = max(contornos, key=cv2.contourArea)
            cv2.drawContours(image, [contorno_principal], -1, (0, 255, 0), 3)

        image_resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        
        
        datax0.append(image_resized)
        datay0.append(count)

    count += 1

n=len(datax0)
M=[]
for i in range(n):
    M+=[i]
random.shuffle(M)


datax1=np.array(datax0)
datay1=np.array(datay0)


trainx0=datax1[M[0:(n//4)*3]]
testx0=datax1[M[(n//4)*3:]]
trainy0=datay1[M[0:(n//4)*3]]
testy0=datay1[M[(n//4)*3:]]

trainy2=to_categorical(trainy0)
testy2=to_categorical(testy0)
X_train=np.array(trainx0).reshape(-1,32,32,3)
Y_train=np.array(trainy2)
X_test=np.array(testx0).reshape(-1,32,32,3)
Y_test=np.array(testy2)

# Aumento de datos
rango_rotacion = 30
mov_ancho = 0.25
mov_alto = 0.25
rango_acercamiento = [0.5, 1.5]

datagen = ImageDataGenerator(
    rotation_range=rango_rotacion,
    width_shift_range=mov_ancho,
    height_shift_range=mov_alto,
    zoom_range=rango_acercamiento,
)

datagen.fit(X_train)


# Modelo
modelo = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(400, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])
# Compilación
modelo.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

modelo.summary()

TAMANO_LOTE = 32
# Entrenar la red
print("Entrenando modelo...");
epocas = 13
history = modelo.fit(
    X_train, Y_train,
    epochs=epocas,
    batch_size=TAMANO_LOTE,
    validation_data=(X_test, Y_test),
    steps_per_epoch=int(np.ceil(len(X_train) / float(TAMANO_LOTE))),
    validation_steps=int(np.ceil(len(X_test) / float(TAMANO_LOTE)))
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epocas)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


print("Modelo entrenado!");
modelo.save('LegoCNN2000.h5') 