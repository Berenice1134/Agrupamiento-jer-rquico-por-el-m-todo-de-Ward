import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
import os
import customtkinter as ctk
from tkinter import filedialog, messagebox
import math

# Creación de la ventana principal
root = ctk.CTk()
root.title("Clasificación Jerárquica")
root.geometry("800x600")  # Define un tamaño inicial para la ventana

# Variables para guardar la ruta del archivo de entrenamiento y las alturas de corte
entrenamiento_archivo = ctk.StringVar()
altura_corte1 = ctk.DoubleVar(value=5.0)
altura_corte2 = ctk.DoubleVar(value=12.0)
altura_corte3 = ctk.DoubleVar(value=20.0)



# Funciones para cargar los archivos y procesar datos
def cargar_archivo_entrenamiento():
    archivo = filedialog.askopenfilename(filetypes=[('Archivos CSV', '*.csv')])
    entrenamiento_archivo.set(archivo)

def mostrar_resultados(resultados_ward):
    for widget in resultados_frame.winfo_children():
        widget.destroy()
    
    scrollable_frame = ctk.CTkScrollableFrame(resultados_frame, width=580, height=380)
    scrollable_frame.pack(pady=20, fill="both", expand=True)
    for paso in resultados_ward:
        texto = f"Agrupamiento de clústeres {paso[0]} y {paso[1]} con distancia de {paso[2]:.2f}. Clústeres restantes: {paso[3]}"
        label = ctk.CTkLabel(scrollable_frame, text=texto)
        label.pack(pady=2, padx=10, fill='x')
    plt.figure(figsize=(12, 9))
    dendrogram(resultados_ward, labels=[str(i+1) for i in range(len(resultados_ward) + 1)])
    plt.title("Dendrograma de Agrupamiento Jerárquico")
    plt.xlabel("Índice de la Muestra")
    plt.ylabel("Distancia de Ward")
    
    # Agregar líneas de corte
    plt.axhline(y=altura_corte1.get(), color='r', linestyle='--')
    plt.axhline(y=altura_corte2.get(), color='b', linestyle='--')
    plt.axhline(y=altura_corte3.get(), color='g', linestyle='--')
    plt.show()

def guardar_clusters_en_excel(resultados_ward, alturas, archivo_nombre_base, datos_originales):
    # Asegurar que el directorio existe
    directorio = os.path.dirname(archivo_nombre_base)
    if not os.path.exists(directorio):
        os.makedirs(directorio)
        
    for altura in alturas:
        clusters = fcluster(resultados_ward, altura, criterion='distance')
        datos_originales[f'Cluster_{altura}'] = clusters
        archivo_nombre = f'{archivo_nombre_base}_altura_{altura}.xlsx'
        
        with pd.ExcelWriter(archivo_nombre, engine='openpyxl') as writer:
            for cluster_num in np.unique(clusters):
                cluster_data = datos_originales[datos_originales[f'Cluster_{altura}'] == cluster_num]
                hoja_nombre = f'Grupo_{cluster_num}'
                cluster_data.to_excel(writer, sheet_name=hoja_nombre, index=False)
                print(f'Datos del grupo {cluster_num} con línea de corte {altura} guardados en el archivo {archivo_nombre} en la hoja {hoja_nombre}')
    

def agrupamiento_ward():
    if entrenamiento_archivo.get():
        datos = pd.read_csv(entrenamiento_archivo.get())
        datos_procesados = preparar_datos(datos)
        X = datos_procesados.values.astype(float)
        resultados_ward = calcular_ward(X, datos)
        mostrar_resultados(resultados_ward)
    else:
        messagebox.showerror("Error", "Por favor, seleccione un archivo primero.")

def preparar_datos(datos):
    return datos.iloc[:, 1:-2]  # Asumiendo que las columnas no numéricas son 'STUDENT ID', 'COURSE ID' y 'GRADE'

def combinaciones(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

def calcular_ward(X, datos_originales):
    # Calcula la matriz de distancias Euclidianas entre cada par de muestras en X
    distancias = np.sqrt(((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=2))
    
    # Establece la diagonal de la matriz de distancias a infinito para evitar considerar la distancia de una muestra consigo misma
    np.fill_diagonal(distancias, np.inf)

    # Número de muestras en el conjunto de datos
    num_muestras = X.shape[0]

    # Inicializa una lista de índices para los clústeres activos, cada muestra comienza como su propio clúster
    clústeres_activos = list(range(num_muestras))

    # Lista para almacenar los resultados de las fusiones de clústeres
    matriz_enlace = []

    # Índice que se asignará a los nuevos clústeres formados por fusiones
    nuevo_indice = num_muestras

    # Array para llevar la cuenta del número de elementos en cada clúster, inicialmente 1 por cada muestra
    tamaños_clusters = np.ones(num_muestras)

    # Continúa fusionando clústeres hasta que solo queda uno
    while len(clústeres_activos) > 1:
        # Encuentra el par de clústeres con la menor distancia entre ellos
        i, j = np.unravel_index(np.argmin(distancias), distancias.shape)
        dist_min = distancias[i, j]

        # Calcula el número de combinaciones posibles
        combinaciones_posibles = combinaciones(len(clústeres_activos), 2)
        print(f"Combinaciones posibles en este paso: {combinaciones_posibles}")

        # Añade el resultado de esta fusión a la matriz de enlace
        matriz_enlace.append([clústeres_activos[i], clústeres_activos[j], dist_min, len(clústeres_activos) - 1])

        # Actualiza el tamaño del clúster resultante de la fusión de i y j
        nuevo_tamaño = tamaños_clusters[i] + tamaños_clusters[j]
        tamaños_clusters[i] = nuevo_tamaño

        # Calcula los centroides de los grupos
        centroid_i = X[[i], :].mean(axis=0)
        centroid_j = X[[j], :].mean(axis=0)

        # Calcula Ek para el grupo resultante de la fusión
        Ek = np.sum((X[[i], :] - centroid_i) ** 2) + np.sum((X[[j], :] - centroid_j) ** 2)
        
        # Calcula ΔE
        np_i = tamaños_clusters[i]
        nq_i = tamaños_clusters[j]
        nt = np_i + nq_i
        delta_E = (np_i * nq_i) / nt

        # Actualiza las distancias del nuevo clúster con todos los otros clústeres
        for k in range(len(distancias)):
            if k != i and k != j:
                dist_ik = distancias[k, i]
                dist_jk = distancias[k, j]
                # Aplica la fórmula de Ward para calcular la nueva distancia entre el clúster fusionado y el clúster k
                distancias[k, i] = distancias[i, k] = np.sqrt(
                    ((np_i + tamaños_clusters[k]) * dist_ik**2 +
                     (nq_i + tamaños_clusters[k]) * dist_jk**2 -
                     tamaños_clusters[k] * dist_min**2) /
                    (np_i + nq_i + tamaños_clusters[k])
                )

        # Elimina el clúster j de la matriz de distancias y del array de tamaños de clústeres
        distancias = np.delete(distancias, j, axis=0)
        distancias = np.delete(distancias, j, axis=1)
        tamaños_clusters = np.delete(tamaños_clusters, j)

        # Actualiza los índices de clústeres activos, asignando el nuevo índice al clúster fusionado
        clústeres_activos[i] = nuevo_indice
        clústeres_activos.pop(j)

        # Incrementa el índice para la próxima fusión
        nuevo_indice += 1

    alturas_corte = [altura_corte1.get(), altura_corte2.get(), altura_corte3.get()]
    guardar_clusters_en_excel(matriz_enlace, alturas_corte, r'C:\Users\bjuar\OneDrive\Documentos\7 semestre\mineria de datos\re', datos_originales)

    # Devuelve la matriz de enlace que describe el orden y las distancias de las fusiones de clústeres
    return np.array(matriz_enlace)

# Widgets de la interfaz
title_label = ctk.CTkLabel(root, text="Clasificación Jerárquica", font=ctk.CTkFont(size=20, weight="bold"))
title_label.pack(pady=20)


entrenamiento_frame = ctk.CTkFrame(root)
entrenamiento_frame.pack(pady=10, padx=10, fill="x")
entrenamiento_label = ctk.CTkLabel(entrenamiento_frame, text="Archivo de Entrenamiento:")
entrenamiento_label.pack(side="left", padx=10)
entrenamiento_archivo_button = ctk.CTkButton(entrenamiento_frame, text="Seleccionar archivo", command=cargar_archivo_entrenamiento)
entrenamiento_archivo_button.pack(pady=20, padx=50, fill='x', expand=True)

alturas_frame = ctk.CTkFrame(root)
alturas_frame.pack(pady=10, padx=10, fill="x")
altura_corte1_label = ctk.CTkLabel(alturas_frame, text="Altura Corte 1:")
altura_corte1_label.pack(side="left", padx=10)
altura_corte1_entry = ctk.CTkEntry(alturas_frame, textvariable=altura_corte1)
altura_corte1_entry.pack(side="left", padx=10)
altura_corte2_label = ctk.CTkLabel(alturas_frame, text="Altura Corte 2:")
altura_corte2_label.pack(side="left", padx=10)
altura_corte2_entry = ctk.CTkEntry(alturas_frame, textvariable=altura_corte2)
altura_corte2_entry.pack(side="left", padx=10)
altura_corte3_label = ctk.CTkLabel(alturas_frame, text="Altura Corte 3:")
altura_corte3_label.pack(side="left", padx=10)
altura_corte3_entry = ctk.CTkEntry(alturas_frame, textvariable=altura_corte3)
altura_corte3_entry.pack(side="left", padx=10)

entrenamiento_button = ctk.CTkButton(root, text="Agrupar", command=agrupamiento_ward)
entrenamiento_button.pack(pady=20, padx=20)

# Frame para mostrar los resultados del agrupamiento
resultados_frame = ctk.CTkScrollableFrame(root)
resultados_frame.pack(pady=20, padx=20, fill="both", expand=True)


root.mainloop()