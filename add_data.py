import sqlite3
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


'''-------------------------------FUNÇÕES-------------------------------'''
def search_db(db_sqlite, query, params=()):
  conn = sqlite3.connect(db_sqlite)
  cursor = conn.cursor()

  cursor.execute(query, params)
  results = cursor.fetchall()

  conn.close()

  return results

def get_advisor_and_student_ids(researcher_id):
  '''Essa função retorna a lista de orientadores e alunos de um pesquisador'''

  # Busca todos os orientadores (advisor_id) que estão associados a títulos acadêmicos pertencentes a um pesquisador específico (researcher_id).
  advisor_ids = search_db(db_sqlite,
                          "SELECT aat.advisor_id\
                           FROM advisors_academic_titles aat\
                           JOIN academic_titles at ON aat.academic_title_id = at.academic_title_id\
                           WHERE at.researcher_id = ?", (researcher_id,))

  # Busca os IDs dos pesquisadores que têm pelo menos um título acadêmico orientado pelo pesquisador especificado (researcher_id)
  student_ids = search_db(db_sqlite,
                          "SELECT r.researcher_id\
                           FROM researchers r\
                           JOIN academic_titles at ON r.researcher_id = at.researcher_id\
                           JOIN advisors_academic_titles aat ON at.academic_title_id = aat.academic_title_id\
                           WHERE aat.advisor_id = ?", (researcher_id,))

  # Corrige a lista
  advisor_ids = [row[0] for row in advisor_ids]
  student_ids = [row[0] for row in student_ids]

  return advisor_ids, student_ids




'''-------------------------------CARREGANDO O BANCO-------------------------------'''

db_sqlite = "./data/mgp.sqlite"

df = pd.read_sql_query('''SELECT
                        r.researcher_id,
                        r.name AS researcher_name,
                        at.academic_title_id,
                        at.title AS academic_title,
                        at.dissertation,
                        at.institution,
                        at.year,
                        at.country,
                        GROUP_CONCAT(aat.advisor_id) AS advisor_ids  -- Lista os orientadores em uma única coluna
                    FROM academic_titles at
                    JOIN researchers r ON r.researcher_id = at.researcher_id
                    LEFT JOIN advisors_academic_titles aat ON at.academic_title_id = aat.academic_title_id
                    GROUP BY at.academic_title_id''', sqlite3.connect(db_sqlite))
sqlite3.connect(db_sqlite).close()







'''-------------------------------GERANDO COLUNA num_students (Número de filhos)-------------------------------'''
###################### AQUI ESCOLHE A QUANTIDADE DE DADOS DO BANCO QUE VAI SER CARREGADA##########################
df_optimized = df.head(50).copy()
#df_optimized = df.copy()

df_optimized["advisor_ids"] = df_optimized["advisor_ids"].apply(lambda x: x.split(",") if x else [])  # transforma os ids do orientadores em uma lista Python
df_optimized["year"] = pd.to_numeric(df_optimized["year"], errors="coerce")                           # converte o ano em número
df_optimized["num_advisors"] = df_optimized["advisor_ids"].apply(len)                                 # Contar quantos orientadores cada título teve e adiciona a coluna

# Calcular o número de alunos por pesquisador
print("Iniciando processamento dos dados...\n")
tqdm.pandas(desc="Calculando número de alunos de um pesquisador")
df_optimized["num_students"] = df_optimized["researcher_id"].progress_apply(lambda rid: len(get_advisor_and_student_ids(rid)[1]))

## Salvando os novos dados
#df_optimized.to_feather("./data/mgp_optimized-num_students.feather")


'''-------------------------------GERANDO COLUNA num_siblings (Número de irmãos)-------------------------------'''
### Listando quantidade de irmãos acadêmicos
# Mapeia orientadores → lista de orientandos
advisor_to_students = defaultdict(set)

for idx, advisors in df_optimized["advisor_ids"].items():
    for advisor in advisors:
        advisor_to_students[advisor].add(idx)

# Agora calcula o número de irmãos para cada pesquisador
def count_siblings(idx, advisors):
    siblings = set()
    for advisor in advisors:
        siblings.update(advisor_to_students[advisor])
    siblings.discard(idx)  # Remove ele mesmo
    return len(siblings)

print("\n")
tqdm.pandas(desc="Calculando número de irmãos")
df_optimized["num_siblings"] = df_optimized.progress_apply(lambda row: count_siblings(row.name, row["advisor_ids"]), axis=1)

## Salvando os novos dados
#df_optimized.to_feather("./data/mgp_optimized.feather")

'''-------------------------------GERANDO COLUNA generation (Distância desde o nó raiz)-------------------------------'''
# Mapeia cada pesquisador para sua lista de orientadores
advisor_map = dict(zip(df_optimized["researcher_id"], df_optimized["advisor_ids"]))

# Armazena os IDs válidos para evitar erros com orientadores ausentes
ids_validos = set(df_optimized["researcher_id"])

# Cache para memoização (melhora performance e evita recálculo)
geracao_cache = {}

ciclos_detectados = []
def cont_geracao(researcher_id, visited=None):
    # Verifica se pesquisador já foi visitado, tratando loop
    if visited is None:
        visited = set()

    
    # Se já calculamos a geração desse pesquisador, retornamos do cache
    if researcher_id in geracao_cache:
        #print("USOU CACHE")
        return geracao_cache[researcher_id]
    
    if researcher_id in visited:
        print(f"Ciclo detectado: {researcher_id}")
        ciclos_detectados.append(researcher_id)
        geracao_cache[researcher_id] = -1  # ou algum valor default
        return -1
    
    visited.add(researcher_id)
    
    orientadores = [int(oid) for oid in advisor_map.get(researcher_id, [])]
    
    # Se não há orientador ou se nenhum deles está no dataset, atribui a geração 0
    if not orientadores or all(oid not in ids_validos for oid in orientadores):
        geracao_cache[researcher_id] = 0
        return 0

    # Calcula a geração de cada orientador
    max_geracao = 0
    for oid in orientadores:
        if oid in ids_validos:
            geracao_orientador = cont_geracao(oid, visited.copy())
            max_geracao = max(max_geracao, geracao_orientador)

    geracao_final = max_geracao + 1
    geracao_cache[researcher_id] = geracao_final
    #print("Geração maior que 0 do pesquisador", researcher_id, ":", geracao_final)
    return geracao_final

# Aplica a função em todos os pesquisadores
print("\n")
tqdm.pandas(desc="Calculando o número das gerações")
df_optimized["generation"] = df_optimized["researcher_id"].progress_apply(cont_geracao)
print("Total de ciclos detectados nas gerações:", len(ciclos_detectados), "\n")


## Salvando os novos dados
df_optimized.to_feather("./data/mgp_optimized.feather")


## Lendo o novo dataframe
df_optimized = pd.read_feather("./data/mgp_optimized.feather")

print("Novo banco:\n", df_optimized.tail(5), "\n\n...tarefa finalizada.\n")
