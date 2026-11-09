# RAG — questions sur le règlement des examens

Mini-projet (BUT Informatique, module *Introduction aux modèles génératifs*) : un assistant qui répond aux
questions des étudiants sur l'organisation des examens, en s'appuyant sur un petit corpus de documents.
Le programme compare deux façons de répondre à la même question :

- **sans RAG** : le modèle répond seul, sans contexte ;
- **avec RAG** (*Retrieval-Augmented Generation*) : on retrouve d'abord les passages pertinents des documents, puis le
  modèle répond à partir d'eux **en citant ses sources** (`[document:numéro_du_passage]`).

Tout tourne en local avec [Ollama](https://ollama.com), sans clé d'API.

## Comment ça marche

1. **Ingestion** (`src/ingest.py`) : lecture des fichiers `.txt` et `.md` du dossier `data/`, découpés en passages
   d'environ 300 mots avec un recouvrement de 50 mots.
2. **Indexation** (`src/rag_chat.py`) : calcul d'un vecteur par passage avec le modèle `nomic-embed-text`, stocké dans
   un index [FAISS](https://github.com/facebookresearch/faiss).
3. **Recherche** : pour une question, les 4 passages les plus proches sont retrouvés.
4. **Génération** : le modèle `llama3` répond en n'utilisant que ces passages, et cite ses sources.

Le rapport complet (choix, tests, limites) est dans [`rapport.md`](rapport.md) (version PDF : [`rapport.pdf`](rapport.pdf)).

## Prérequis

- [Ollama](https://ollama.com) installé et lancé
- Python **inférieur à 3.12**
- Les deux modèles :

```bash
ollama pull nomic-embed-text
ollama pull llama3
```

## Installation

Linux / macOS :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell) :

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Puis, dans les deux cas :

```bash
pip install --upgrade pip
pip install faiss-cpu numpy requests
```

## Utilisation

1. **Construire l'index** (à refaire quand les documents de `data/` changent) :

```bash
python -c "from src.rag_chat import build_index; build_index()"
```

2. **Générer le tableau comparatif sans RAG / avec RAG** sur des questions prédéfinies (dont deux hors des documents) :

```bash
python -m tests.test_questions
```

Le résultat est écrit dans `tests/comparaison_noRAG_vs_RAG.md`.

3. *(Optionnel)* **Poser vos propres questions** :

```bash
python -m tests.own_questions
```

## Structure

```
data/      documents sources (règlement, FAQ, surveillance, rattrapage)
src/       ingestion et découpage (ingest.py), index, recherche et génération (rag_chat.py)
tests/     comparaison sans RAG / avec RAG, questions libres
rapport.md rapport du projet (aussi en PDF)
```

## Licence

Code sous [licence MIT](LICENSE).
