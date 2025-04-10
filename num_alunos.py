import sqlite3
import joblib   # Armazenar e carregar modelo treinado

import numpy as np
import pandas as pd

from tqdm import tqdm
from pyoperon.sklearn import SymbolicRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error


'''-------------------------------FUNÇÕES-------------------------------'''




'''-------------------------------CARREGANDO O BANCO-------------------------------'''

df_optimized = pd.read_feather("./data/mgp_optimized.feather")

#df_optimized = df_optimized.head(200)
#print(len(df_optimized))


'''-------------------------------TREINANDO O MODELO-------------------------------'''

# Selecionar as colunas relevantes para prever o número de alunos
features = ["institution", "year", "country", "num_advisors"]
target = "num_students"

# Remover valores nulos
print("\n\nTamanho do banco com valores nulos:", len(df_optimized))
df_optimized = df_optimized.dropna(subset=features)
print("Novo tamanho do banco sem valores nulos:", len(df_optimized))


# Codificar variáveis categóricas (institution, country)
label_encoders = {}
for col in ["institution", "country"]:
    le = LabelEncoder()
    df_optimized[col] = le.fit_transform(df_optimized[col].str.strip().str.lower().str.replace(" ", ""))
    label_encoders[col] = le

# Normalizar valores numéricos
#scaler = MinMaxScaler()
#df_optimized[features] = scaler.fit_transform(df_optimized[features])

# Converter em Fortran-order
features_df = np.asfortranarray(df_optimized[features].values, dtype=np.float64)
target_df = np.asfortranarray(df_optimized[target].values, dtype=np.float64)

# Criar o regressor simbólico
model = SymbolicRegressor(
    n_threads=31
)

# Treinar o modelo com os dados
#model.fit(df_optimized[features], df_optimized[target])
print("\n\nTreinamento iniciado...\n")
model.fit(features_df, target_df)

# Exibir o melhor modelo encontrado
print("Melhor expressão encontrado:\n", model.pareto_front_[0]['model'])

'''-------------------------------SALVANDO O MODEL-------------------------------'''

# Salvar o modelo treinado
#joblib.dump(model, "./data/modelo_treinado.pkl")

print("\n\n...Treinamento finalizado.")



'''-------------------------------AVALIANDO O MODEL-------------------------------'''

# Fazer previsões
df_optimized["predicted_students"] = model.predict(np.asfortranarray(df_optimized[features].values, dtype=np.float64))

# Visualizar algumas previsões
print("\n\nVisualização de algumas predições:\n", df_optimized[[target, "predicted_students"]].head(10))

# Calcular o erro absoluto
mae = mean_absolute_error(df_optimized[target], df_optimized["predicted_students"])
print("\nErro médio absoluto:", mae)


'''-------------------------------TESTANDO UMA PREDIÇÃO-------------------------------'''

while True:
  # Perguntar sobre um novo pesquisador
  print("\n\n")
  input_institution = input("Instituição: ")
  input_year = int(input("Ano: "))
  input_country = input("País: ")
  input_num_advisors = int(input("Número de orientadores: "))


  researcher_new = {
      "institution": input_institution,
      "year": input_year,
      "country": input_country,
      "num_advisors": input_num_advisors
      }


  # Aplicando a transformação aos inputs categóricos
  for col in ["institution", "country"]:
      researcher_new[col] = researcher_new[col].strip().lower().replace(" ", "")
      if researcher_new[col] in label_encoders[col].classes_:
          researcher_new[col] = int(label_encoders[col].transform([researcher_new[col]])[0])
      else:
          print(f"Aviso: '{researcher_new[col]}' não está nos dados de treinamento. Usando valor padrão -1.")
          researcher_new[col] = -1  # Valor para desconhecidos

  print("Dados de entrada:", researcher_new)
  researcher_fortran = np.asfortranarray([[researcher_new[col] for col in features]], dtype=np.float64)

  print("Pesquisador exemplo deve ter:", max(1, round(model.predict(researcher_fortran)[0])), "aluno(s)")

  print("\n\n")
  repetir = input("Deseja fazer outra pergunta? (s/n): ")
  if repetir != 's':
    break
