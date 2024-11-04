# Chatbot RAG com Streamlit

Este é um chatbot RAG (Retrieval-Augmented Generation) que utiliza LangChain e Streamlit, com uma base de dados vetorial para recuperação de respostas e o modelo de linguagem Mistral-7B da ChatMistralAI para reformulação de perguntas e geração de respostas.


## Pré-requisitos

1. **Python 3.11.10**.
2. **Instalar as dependências**: Todas as bibliotecas e pacotes necessários estão listados no arquivo `requirements.txt`.

## Instruções de instalação

1. **Instalar os requisitos**:
Crie um ambiente virtual e ative-o:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Para Linux/Mac
    .\venv\Scripts\activate  # Para Windows
     ```

2. **Instalar dependências**:
    ```bash
    pip install -r requirements.txt
     ```

Para iniciar o chatbot com a interface Streamlit, execute o seguinte comando no terminal:
```bash
streamlit run chatbot_ui.py
 ```


Após executar este comando, a interface do chatbot será aberta no seu navegador padrão.

