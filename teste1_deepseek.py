import requests

# Defina a URL do servidor vLLM rodando localmente
API_URL = "http://localhost:8123/v1/chat/completions"

# Corpo da requisição no formato OpenAI
payload = {
    "model": "TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ",
    "messages": [
        {"role": "system", "content": "Você é um assistênte útil e educado."},
        {"role": "user", "content": "O que são LLMs?"}
    ],
    "max_tokens": 400,
    "temperature": 0.7
}

# Cabeçalhos HTTP
headers = {"Content-Type": "application/json"}

# Faz a requisição para a API
response = requests.post(API_URL, json=payload, headers=headers)

# Exibe a resposta formatada
if response.status_code == 200:
    resposta = response.json()
    print("\n🤖 Resposta do LLM:\n\n")
    print(resposta["choices"][0]["message"]["content"])
    print("\n\n🤖 Fim da respostas do LLM.\n")
else:
    print("Erro:", response.status_code, response.text)

