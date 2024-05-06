import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Recopilar y preprocesar datos
data = pd.read_csv('CSV/emails.csv')  # Suponiendo que los datos se almacenan en un CSV
X = data['contenido']  # Extraemos el contenido de los correos electrónicos
y = data['categoria']  # Obtenemos las categorías correspondientes

# Vectorización TF-IDF, para convertir el texto de los correos
# electrónicos en características numéricas que puedan ser utilizadas por el modelo de clasificación.
vectorizer = TfidfVectorizer()
X_vectorizado = vectorizer.fit_transform(X)  # Convertimos el texto en características numéricas

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_vectorizado, y, test_size=0.2)

# Entrenar el modelo Naive Bayes
modelo = MultinomialNB()
modelo.fit(X_train, y_train)

# Evaluar el rendimiento del modelo
predicciones = modelo.predict(X_test)

# Calcular métricas de evaluación

# Precisión, que mide la proporción de correos electrónicos clasificados como "spam" que realmente son "spam".
precision = precision_score(y_test, predicciones, pos_label='spam')
# Revocación, que mide la proporción de correos electrónicos "spam" que fueron correctamente clasificados como "spam".
recall = recall_score(y_test, predicciones, pos_label='spam')
# Puntaje F1, que es una medida combinada de precisión y revocación.
f1 = f1_score(y_test, predicciones, pos_label='spam')
# Precisión general, que mide la proporción de correos electrónicos clasificados correctamente (ya sea "spam" o
# "principal").
accuracy = accuracy_score(y_test, predicciones)

print(f"Precisión: {precision:.3f}")
print(f"Revocación: {recall:.3f}")
print(f"Puntaje F1: {f1:.3f}")
print(f"Precisión general: {accuracy:.3f}")

while True:
    nuevo_correo = input("Ingrese el correo: ")

    if nuevo_correo.lower() == 'c':
        break

    # Vectorizar el nuevo correo y realizar la predicción
    nuevo_correo_vectorizado = vectorizer.transform([nuevo_correo])
    prediccion_nuevo_correo = modelo.predict(nuevo_correo_vectorizado)

    if prediccion_nuevo_correo[0] == 'principal':
        print(f"El nuevo correo electrónico se clasifica como 'principal'.")
    elif prediccion_nuevo_correo[0] == 'spam':
        print(f"El nuevo correo electrónico se clasifica como 'spam'.")
    else:
        print(f"Categoría no reconocida: {prediccion_nuevo_correo[0]}")
