# deep-learning-lego-image-classification
Clasificación de piezas LEGO mediante CNN con análisis comparativo de técnicas de preprocesamiento usando TensorFlow y OpenCV.

# Clasificación de LEGO con CNN – Proyecto de Visión por Computadora

Proyecto de Deep Learning enfocado en la clasificación de imágenes de piezas de LEGO utilizando Redes Neuronales Convolucionales (CNN) y técnicas avanzadas de preprocesamiento de imágenes.

# Resumen del Proyecto

Este proyecto implementa y evalúa diferentes pipelines de preprocesamiento y arquitecturas CNN para la clasificación de imágenes utilizando el Conjunto de Datos de Clasificación LEGO B200C (Kaggle).

El objetivo principal es analizar cómo las técnicas de mejora de imágenes afectan el rendimiento y la generalización del modelo.

# Conjunto de Datos

- Conjunto de Datos: Conjunto de Datos de Clasificación LEGO B200C
- Clases utilizadas: 4 categorías seleccionadas de piezas LEGO
- Imágenes por clase: 4000
- Tamaño de imagen: 64x64 (redimensionadas a 32x32 para los experimentos)
- Formato: RGB (.jpg)
- Conjunto de datos equilibrado

# Técnicas de Preprocesamiento Evaluadas

Se implementaron y compararon dos flujos de preprocesamiento:

# Método 1 – Desenfoque + Detección de Bordes Canny
- Normalización
- Desenfoque Gaussiano (3x3)
- Conversión a escala de grises
- Detección de bordes Canny
- Fusión de bordes con la imagen original
- Redimensionar a 32x32

# Método 2 – Gaussiano + Otsu + Contornos
- Conversión a escala de grises
- Desenfoque gaussiano (5x5)
- Umbralización de Otsu
- Detección de contornos
- Contorno más grande dibujado sobre la imagen original
- Redimensionar a 32x32

# Arquitectura CNN

El modelo CNN implementado consiste en:

- Conv2D (32 filtros, 3x3, ReLU)
- MaxPooling (2x2)
- Conv2D (64 filtros, 3x3, ReLU)
- MaxPooling (2x2)
- Dropout (0.5)
- Aplanar
- Densa (400 neuronas, ReLU)
- Densa (4 neuronas, Softmax)

Optimizador: Adam
Función de Pérdida: Entropía Cruzada Categórica
Métrica: Precisión  

# Resultados Experimentales

Se realizaron diferentes experimentos variando:

- Número de clases
- Número de imágenes
- Tamaño de la imagen (32x32 / 64x64)
- Épocas
- Técnica de preprocesamiento

# Mejor Modelo Equilibrado
- Tamaño de imagen: 32x32
- Épocas: 13
- Preprocesamiento: Gaussiano + Otsu + Contornos
- Precisión en entrenamiento: ~86%
- Precisión en validación: ~85%
- Menor sobreajuste en comparación con el modelo base

# Principales Hallazgos

- Las técnicas de preprocesamiento afectan significativamente la generalización del modelo.
- El realce basado en bordes mejora la extracción de características pero puede aumentar el tiempo de entrenamiento.
- Un número excesivo de épocas puede llevar a sobreajuste.
- Un conjunto de datos equilibrado contribuye a una precisión de validación estable.

Este programa puede modificarse para obtener los datos directamente desde Kaggle en lugar de almacenarlos localmente. Para ello, es posible utilizar el dataset en línea mediante la Kaggle API oficial, la cual permite descargar y gestionar datasets directamente desde la PC.

La Kaggle API proporciona un mecanismo de autenticación mediante un archivo kaggle.json, que se obtiene desde la configuración de la cuenta en Kaggle. Una vez configurada, el programa puede automatizar la descarga del dataset antes de ejecutar el entrenamiento del modelo, evitando así la necesidad de mantener los datos manualmente en el equipo local.

De esta manera, el proyecto se vuelve más portable, reproducible y profesional, ya que cualquier persona puede ejecutar el código y descargar automáticamente el dataset desde Kaggle sin depender de archivos locales preexistentes.

# Cómo Ejecutar

```bash
python main.py -p "path_to_dataset"

