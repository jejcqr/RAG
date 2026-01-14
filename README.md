# PRÉREQUIS

## 1) Installer Ollama depuis https://ollama.com
## 2) Puis dans un terminal :

ollama pull nomic-embed-text

ollama pull llama3

## POUR WINDOWS :

python -m venv .venv

.\.venv\bin\activate

## POUR LINUX : 

python3 -m venv .venv

source .venv/bin/activate

# ENVIRONNEMENT

pip install --upgrade pip

pip install faiss-cpu numpy requests

# UTILISATION

## 1) Construire l’index
python -c "from src.rag_chat import build_index; build_index()"

## 2) Générer le tableau comparatif no-RAG vs RAG
python -m tests.test_questions

## (Optionnel) 3) Si vous voulez poser vos propres questions
python -m tests.own_questions

