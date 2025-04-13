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

df_optimized = df.head(50).copy()
#df_optimized = df.copy()

df_optimized["advisor_ids"] = df_optimized["advisor_ids"].apply(lambda x: x.split(",") if x else [])  # transforma os ids do orientadores em uma lista Python
df_optimized["year"] = pd.to_numeric(df_optimized["year"], errors="coerce")                           # converte o ano em número
df_optimized["num_advisors"] = df_optimized["advisor_ids"].apply(len)                                 # Contar quantos orientadores cada título teve e adiciona a coluna

# Calcular o número de alunos por pesquisador
tqdm.pandas(desc="Calculando número de alunos")
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

df_optimized["num_siblings"] = df_optimized.apply(lambda row: count_siblings(row.name, row["advisor_ids"]), axis=1)

## Salvando os novos dados
df_optimized.to_feather("./data/mgp_optimized.feather")


## Lendo o novo dataframe
df_optimized = pd.read_feather("./data/mgp_optimized.feather")

print("Novo banco:\n", df_optimized.tail(5), "\n\n...tarefa finalizada.")
