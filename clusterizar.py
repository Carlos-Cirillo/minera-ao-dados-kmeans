import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.datasets import load_files

# --- 1. Carregar os Dados ---

dataset_path = 'C:/Users/andra/Downloads/reuter+50+50/C50train'

# O Scikit-learn lê as subpastas (autores) como categorias
# e carrega os arquivos de texto
try:
    # 'latin1' é um encoding comum para essa base de dados antiga
    dataset = load_files(dataset_path, encoding='latin1')
except FileNotFoundError:
    print(f"Erro: Caminho não encontrado: '{dataset_path}'")
    print("Por favor, baixe a base Reuters (C50) e ajuste o 'dataset_path' no script.")
    exit()

print(f"Arquivos carregados: {len(dataset.data)}")

# --- 2. Vetorização (Equivalente ao seq2sparse + TF-IDF) ---
# Converte os textos em uma matriz de números (vetores TF-IDF)
vectorizer = TfidfVectorizer(
    max_df=0.5,           # Ignora palavras muito frequentes
    min_df=2,             # Ignora palavras muito raras
    stop_words='english'  # Remove palavras comuns (ex: 'the', 'is')
)
X = vectorizer.fit_transform(dataset.data)

print("Textos vetorizados.")

# --- 3. Executar o K-Means (Equivalente ao mahout kmeans) ---
# Vamos usar k=50, pois a base C50 tem 50 autores (clusters)
k = 10 

model = KMeans(
    n_clusters=k,
    init='k-means++',
    max_iter=100, # Similar ao -x 10 do mahout
    n_init=1      # Roda o algoritmo uma vez
)
model.fit(X)

print(f"Clusterização K-Means (k={k}) completa.")

# --- 4. Analisar os Resultados (Equivalente ao clusterdump) ---
print("\n--- Resultado dos Clusters (Top 10 termos por cluster) ---")

# Pega os nomes das "palavras" (features)
terms = vectorizer.get_feature_names_out()

# Pega os "centros" de cada cluster
order_centroids = model.cluster_centers_.argsort()[:, ::-1]

# Itera em cada cluster (de 0 a k-1)
for i in range(k):
    print(f"Cluster {i}:")
    
    # Pega as 10 palavras mais importantes (top terms)
    top_terms = [terms[ind] for ind in order_centroids[i, :10]]
    print(f"  {', '.join(top_terms)}")

print("\nAnálise concluída.")