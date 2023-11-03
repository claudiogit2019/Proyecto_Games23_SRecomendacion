from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = FastAPI()

# Cargar el DataFrame
df_juegos = pd.read_csv('dfx_merge_sistema.csv')

# Combinar las características relevantes
df_juegos['combined_features'] = df_juegos['Genero'] + ' ' + df_juegos['Etiquetas'] + ' ' + df_juegos['Especificaciones']

# Inicializar el vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer()

# Calcular la matriz TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(df_juegos['combined_features'])

# Calcular la similitud coseno
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

@app.get('/recomendacion_juego/{Id}')
def recomendacion_juego(Id: int):
    try:
        idx = df_juegos[df_juegos['Id'] == Id].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Ordenar por similitud y excluir el juego original
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [x for x in sim_scores if x[0] != idx]

        # Elegir las primeras 5 recomendaciones, evitando duplicados
        recommended_juegos = []
        seen_ids = set()
        for i in sim_scores:
            juego_id = i[0]
            if juego_id not in seen_ids and i[1] < 0.3:  # Establece un umbral de similitud
                recommended_juegos.append(df_juegos.iloc[juego_id]['Nombre_del_contenido'])
                seen_ids.add(juego_id)
            if len(recommended_juegos) >= 5:
                break

        return recommended_juegos[:5]  # Limita el resultado a 5 recomendaciones
    except IndexError:
        return {"message": "Juego no encontrado"}
