import pandas as pd
from fastapi import FastAPI, HTTPException
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

app = FastAPI()


#///////////////////////////////////////////////////////////////////////////////////////////////////
df_juegos = pd.read_csv('dfx_merge_sistema.csv')

    # Combinar 
df_juegos['combined_features'] = df_juegos['Genero'] + ' ' + df_juegos['Etiquetas'] + ' ' + df_juegos['Especificaciones'] 

    # Asegurar
df_juegos['combined_features'] = df_juegos['combined_features'].apply(lambda x: ' '.join(x.split()))

    # Matriz
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(df_juegos['combined_features'])

    # Similitud 
cosine_sim = cosine_similarity(count_matrix, count_matrix)

@app.get('/recomendacion_juego/{Id}')

def recomendacion_juego(Id: int):
    try:
        
        idx = df_juegos[df_juegos['Id'] == Id].index[0]

        
        sim_scores = list(enumerate(cosine_sim[idx]))

        
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        
        top_juegos = sim_scores[1:6]

        
        recommended_juegos = [df_juegos.iloc[i[0]]['Nombre_del_contenido'] for i in top_juegos]

        return recommended_juegos
    except IndexError:
        return {"message": "Juego no encontrado"}
