import os
from openai import OpenAI

def generate_sql_query(user_query):
    '''Essa função usa IA para converter pergunta em SQL parametrizado.'''

    system_prompt = """Você é um assistente especializado em bancos de dados SQLite. Converta perguntas em SQL parametrizado usando `?` para os valores.

    Estrutura do banco que você é especialista:
    researchers (
      researcher_id INTEGER PRIMARY KEY,
      name TEXT NOT NULL
    )

    academic_titles (
      academic_title_id INTEGER PRIMARY KEY,
      researcher_id INTEGER NOT NULL,
      title TEXT NOT NULL,
      dissertation TEXT,
      institution TEXT,
      year INTEGER,
      country TEXT,
      FOREIGN KEY (researcher_id) REFERENCES researchers (researcher_id) ON DELETE CASCADE
    )

    advisors_academic_titles (
      advisor_id INTEGER NOT NULL,
      academic_title_id INTEGER NOT NULL,
      PRIMARY KEY (advisor_id, academic_title_id),
      FOREIGN KEY (advisor_id) REFERENCES researchers (researcher_id) ON DELETE CASCADE,
      FOREIGN KEY (academic_title_id) REFERENCES academic_titles (academic_title_id) ON DELETE CASCADE
    )

    Exemplo de saída correta:
    ['SELECT name FROM researchers WHERE researcher_id = ?', ['9', '22']]

    Retorne apenas a SQL e os valores em uma lista Python, sem explicações. Não inclua comandos DELETE ou UPDATE.
    """

    response = client.chat.completions.create(
        model="TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )

    return eval(response.choices[0].message.content.strip())



client = OpenAI(
	base_url="http://localhost:8123/v1",
	api_key="so_para_nao_ficar_vazio"
)

def chatbot(user_query):
    sql_data = generate_sql_query(user_query)
    print("🔍 Query gerada:", sql_data)

    try:
        results = search_db(db_sqlite,sql_data[0], sql_data[1])
        if results:
            return results
        return "Nenhum resultado encontrado."
    except Exception as e:
        return f"Erro ao executar a query: {e}"

# Testando
user_input = "Qual o total de orientadores?"
print(chatbot(user_input))
