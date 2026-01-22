# Job Market Explorer

**Job Market Explorer** est une application Streamlit qui permet d’explorer, analyser et comparer le marché de l’emploi data en France à partir de sources publiques et privées (France Travail, Welcome to the Jungle).

L’objectif est de transformer des annonces en insights exploitables : compétences demandées, salaires, localisations, types de contrats, et adéquation avec un profil candidat.

---

## Fonctionnalités principales

### Collecte des offres
- **France Travail** (API officielle)
- **Welcome to the Jungle** (scraping contrôlé)

### Analyse du marché
- Volume d’offres
- Répartition géographique
- Types de contrats (CDI, etc.)
- Présence ou non de salaire

### Analyse des compétences
- Hard skills (requis / optionnels)
- Soft skills
- Détection de compétences émergentes

### Analyse des salaires
- % d’offres avec salaire
- Distribution (lorsque exploitable)

### Matching CV vs Offres
- Matching par compétences
- Matching sémantique 
- Scoring global par offre

### Visualisation géographique
- Carte des offres (France Travail uniquement)

---

## Stack technique

- **Python**
- **Streamlit** (UI & déploiement)
- **Pandas**
- **APIs REST** (France Travail)
- **Web scraping** (WTTJ)
- **NLP / embeddings** (matching sémantique)
- **Git / GitHub**

---


## Installation & exécution

### 1. Cloner le projet
```bash
git clone https://github.com/linh-dinh-1012/job-market-explorer.git

### 2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate

### 3. Installer les dépendances
```bash
pip install -r requirements.txt

4. Lancer l’application
```bash
streamlit run app.py

