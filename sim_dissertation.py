import pandas as pd
import numpy as np

from tqdm import tqdm
from pyoperon.sklearn import SymbolicRegressor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

'''-------------------------------CARREGANDO O BANCO-------------------------------'''

df_optimized = pd.read_feather("./data/mgp_optimized.feather")

df_optimized = df_optimized.head(2000)
print(len(df_optimized))


'''-------------------------------TREINANDO O MODELO-------------------------------'''

# 1️⃣ Carregar modelo BERT
bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')

# 2️⃣ Criar embeddings dos orientadores
print("\n\nTamanho do banco com valores nulos:", len(df_optimized))
df_sem_nulos = df_optimized["dissertation"].dropna()
print("Novo tamanho do banco sem valores nulos:", len(df_sem_nulos))
orientadores_emb = {idx: bert_model.encode(text) for idx, text in tqdm(df_sem_nulos.items(), desc="Gerando Embeddings")}

# 3️⃣ Criar DataFrame de treinamento
dados = []
for nome, emb in orientadores_emb.items():
    features = emb  # Embedding BERT como features
    dados.append([nome, *features])

df_train = pd.DataFrame(dados, columns=["orientador"] + [f"feat_{i}" for i in range(len(features))])

# 4️⃣ Criar similaridade como alvo (usamos o próprio cosseno como target para ensinar o modelo)
sim_matrix = cosine_similarity(np.array(list(orientadores_emb.values())))
df_train["similarity"] = sim_matrix.diagonal()  # Similares com elas mesmas (idealmente, dados rotulados viriam de alunos reais)

# 5️⃣ Treinar modelo de Regressão Simbólica
X = df_train.drop(columns=["orientador", "similarity"]).values  # Features (Embeddings)
y = df_train["similarity"].values  # Similaridade como target

model = SymbolicRegressor(
    n_threads=31
)

print("\n\nTreinamento iniciado...\n")
model.fit(X, y)

print("...treinamento finalizado.")

# 6️⃣ Função para encontrar o melhor orientador para uma nova dissertação
def melhor_orientador(nova_dissertacao):
    nova_emb = bert_model.encode(nova_dissertacao).reshape(1, -1)  # Gerar embedding
    pred_sim = model.predict(nova_emb)[0]  # Calcular similaridade

    # Comparar com todos os orientadores
    melhor_nome = max(orientadores_emb.keys(), key=lambda nome: cosine_similarity([orientadores_emb[nome]], nova_emb)[0, 0])

    return melhor_nome, pred_sim
    

'''-------------------------------TESTANDO UMA PREDIÇÃO-------------------------------'''

while True:
  input_dissertacao = input("\n\nDigite a dissertação: ")
  # 7️⃣ Usando com uma dissertação
  melhor, score = melhor_orientador(input_dissertacao)

  melhor_researcher = df_optimized.iloc[melhor]
  #print(f"Melhor orientador para essa dissertação: {melhor_researcher} (similaridade: {score:.3f})")
  print(f"\n\n🚀 Melhor orientador para essa dissertação 🚀\nID MGP: {melhor_researcher['researcher_id']}\nNome: {melhor_researcher['researcher_name']}\nDissertação: {melhor_researcher['dissertation']} (similaridade: {score:.2%})")

  repetir = input("\n\nDeseja repetir? (s/n): ")
  if repetir.lower() != "s":
    break
