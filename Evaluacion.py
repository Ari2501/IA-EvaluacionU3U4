import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore # Aumento de datos
import os

# Para procesar mis imágenes
def preprocess_image(image_path, target_size=(64, 64)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró la imagen en la ruta: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image / 255.0  

# Cargar mis imágenes y etiquetas
def load_dataset(image_paths, labels, target_size=(64, 64)):
    images = []
    valid_labels = []
    
    for img_path, label in zip(image_paths, labels):
        try:
            img = preprocess_image(img_path, target_size)
            images.append(img)
            valid_labels.append(label)  
        except Exception as e:
            print(f"Error procesando la imagen {img_path}: {e}")
    
    return np.array(images), np.array(valid_labels)

# Lista de mis rutas y etiquetas
image_paths = [
    r'C:/Users/Aranz/Downloads/IA EV U3U4/Taj Mahal.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/TM2.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/TM3.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/TM4.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/TM5.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/Muralla China.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/MCH2.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/MCH3.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/MCH4.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/MCH5.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/Chichen Itza.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/CHI2.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/CHI3.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/CHI4.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/CHI5.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/Burj Khalifa.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/BK2.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/BK3.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/BK4.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/BK5.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/Monte Rushmore.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/MR2.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/MR3.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/MR4.jpg',
    r'C:/Users/Aranz/Downloads/IA EV U3U4/MR5.jpg'
]

# Las etiquetas con longitud igual a la de image_paths
labels = [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5  # 5 imágenes por cada monumento que puse

# Carga mis datos y modifica
X, y = load_dataset(image_paths, labels)
if len(X) == 0:
    raise ValueError("No se cargaron imágenes correctamente. Revisa las rutas.")
    
y = to_categorical(y, num_classes=5)  # 5 clases

# Datos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Aquí se crea el modelito
model = Sequential([
    Flatten(input_shape=(64, 64, 3)),  # Tamaño = (64x64 con 3 canales RGB)
    Dense(128, activation='relu'),
    Dropout(0.3),  # Regularización con dropout
    Dense(64, activation='relu'),
    Dropout(0.3),  # Regularización con dropout
    Dense(5, activation='softmax')  # Salida para 5 clases
])

# Aquí compila el modelito
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Configuración de aumento de datos (Data Augmentation)
datagen = ImageDataGenerator(
    rotation_range=30,  # se rotan imágenes aleatoriamente
    width_shift_range=0.2,  # se trasladan las imágenes horizontalmente
    height_shift_range=0.2,  # se trasladan las imágenes verticalmente
    shear_range=0.2,  # corta imágenes aleatoriamente
    zoom_range=0.2,  # se acercan aleatoriamente las imágenes
    horizontal_flip=True,  # voltea imágenes horizontalmente
    fill_mode='nearest'  # se rellenan píxeles vacíos después de las transformaciones
)

# aquí ajusta el modelo con aumento de datos
datagen.fit(X_train)

# aquí entrena el modelo con aumento de datos
model.fit(datagen.flow(X_train, y_train, batch_size=16), validation_data=(X_val, y_val), epochs=20)  # Aumenté las épocas

# aquí es para guardar el modelo
model.save('my_model_with_augmentation.keras')

# aquí evalúa el modelito
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Precisión en conjunto de validación: {accuracy*100:.2f}%")

# mis etiquetas para las predicciones
class_names = ["Taj Mahal", "Muralla China", "Chichen Itza", "Burj Khalifa", "Monte Rushmore"]

# aquí para la predicción usando la cámara web
def predict_from_webcam(model, target_size=(64, 64)):
    cap = cv2.VideoCapture(0)  #abre la camarita
    if not cap.isOpened():
        print("No se puede acceder a la cámara.")
        return
    
    capture_count = 0  # contador para las imágenes que capturo
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se puede leer la imagen desde la cámara.")
            break
        
        # se preprocesa la imagen
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, target_size) / 255.0  
        img_expanded = np.expand_dims(img_resized, axis=0)  
        
        # Predicción
        predictions = model.predict(img_expanded)
        class_idx = np.argmax(predictions)
        predicted_class = class_names[class_idx]
        
        # aquí muestra el resultado en la imagen
        cv2.putText(frame, f"Se predice que es: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # aquí muestra la imagen con la predicción
        cv2.imshow('Cámara Web', frame)
        
        # este para capturar mi imagen al presionar 'z'
        if cv2.waitKey(1) & 0xFF == ord('z'):
            capture_count += 1
            image_name = f"captura_{capture_count}.jpg"
            cv2.imwrite(image_name, frame)  
            print(f"Imagen capturada y guardada como {image_name}")
        
        # para salir de la camara se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# aquí llama a la función para predecir usando la cámara web
predict_from_webcam(model)
