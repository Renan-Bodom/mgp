# MGP

Este projeto utiliza dados do Mathematics Genealogy Project (MGP) para análise de genealogia acadêmica, exploração de relações orientador-aluno e aplicação de técnicas de inteligência artificial, incluindo regressão simbólica e embeddings de linguagem.

## Estrutura do Projeto

- **IA no MGP.ipynb**: Notebook principal com análises, visualizações, modelagem preditiva e recomendações de orientadores.
- **Raspagem MGP**: Notebook com códigos para executar a raspagem dos dados.
- **data/**: Dados processados, banco SQLite, arquivos Feather e encoders.
- **add_data.py**: Scripts para processamento, limpeza e enriquecimento dos dados do MGP.
- **mgp_with_deepseek.py**: Geração de queries SQL a partir de linguagem natural usando IA.
- **Teste PyOperon.ipynb**: Exemplos de uso da regressão simbólica com PyOperon.
- **Pesquisadores com ciclos.txt**: IDs de pesquisadores com ciclos detectados na árvore genealógica.

## Principais Funcionalidades

- Consulta e análise de relações acadêmicas (orientadores, alunos, gerações).
- Previsão do número de alunos orientados com regressão simbólica.
- Recomendação de potenciais orientadores baseada em similaridade de dissertações (embeddings BERT).
- Geração automática de queries SQL parametrizadas via IA.
- Visualização de distribuições e relações acadêmicas.

## Como usar

1. Instale as dependências:
   ```sh
   pip install -r requirements.txt
   ```

2. Abra o notebook "IA no MGP.ipynb" para manipular e analisar os dados.