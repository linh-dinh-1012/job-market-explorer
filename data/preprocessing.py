import pandas as pd

# ==================================================
# HELPERS
# ==================================================
def safe_list(x):
    """Ensure value is a list"""
    return x if isinstance(x, list) else []


def extract_required_optional(items):
    """
    Extract required (E) and optional (S) labels from FT-style lists
    """
    if not isinstance(items, list):
        return [], []

    required = [i.get("libelle") for i in items if i.get("exigence") == "E"]
    optional = [i.get("libelle") for i in items if i.get("exigence") == "S"]

    return required, optional


# ==================================================
# WELCOME TO THE JUNGLE
# ==================================================
def preprocess_wttj(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Welcome to the Jungle offers
    Note: WTTJ has NO notion of required/optional
    """

    return pd.DataFrame({
        "id": df.index.astype(str),
        "title": df.get("title"),
        "description": df.get("description"),
        "company": df.get("company"),
        "location": df.get("location"),
        "contract": df.get("contract"),
        "salary": df.get("salary"),
        "date": df.get("date"),
        "industry": df.get("industry_category"),
        "experience": df.get("experience"),
        "education": df.get("education"),

        # -------- SKILLS --------
        # Hard skills: techniques + savoir-faire (non hiérarchisés)
        "skills_hard_required": (
            df.get("competences_techniques", []).apply(safe_list) +
            df.get("savoir_faire", []).apply(safe_list)
        ),
        "skills_hard_optional": [[] for _ in range(len(df))],

        # Soft skills
        "skills_soft": df.get("savoir_etre").apply(safe_list),

        # -------- LANGUAGES --------
        "languages_required": [[] for _ in range(len(df))],
        "languages_optional": df.get("langues").apply(safe_list),

        "source": "Welcome to the Jungle",
        "url": df.get("link"),
    })


# ==================================================
# FRANCE TRAVAIL
# ==================================================
def preprocess_ft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize France Travail offers (E/S respected)
    """

    skills_required = []
    skills_optional = []
    languages_required = []
    languages_optional = []

    for _, row in df.iterrows():
        # Hard skills
        r, o = extract_required_optional(row.get("competences"))
        skills_required.append(r)
        skills_optional.append(o)

        # Languages
        lr, lo = extract_required_optional(row.get("langues"))
        languages_required.append(lr)
        languages_optional.append(lo)

    return pd.DataFrame({
        "id": df.get("id"),
        "title": df.get("intitule"),
        "description": df.get("description"),
        "company": df.get("entreprise.nom"),
        "location": df.get("lieuTravail.libelle"),
        "contract": df.get("typeContratLibelle"),
        "salary": df.get("salaire.libelle"),
        "date": df.get("dateCreation"),
        "industry": df.get("secteurActiviteLibelle"),
        "experience": df.get("experienceLibelle"),
        "education": df.get("formations"),

        # -------- SKILLS --------
        "skills_hard_required": skills_required,
        "skills_hard_optional": skills_optional,

        # Soft skills (no E/S)
        "skills_soft": df.get("qualitesProfessionnelles").apply(
            lambda x: [q.get("libelle") for q in x] if isinstance(x, list) else []
        ),

        # -------- LANGUAGES --------
        "languages_required": languages_required,
        "languages_optional": languages_optional,

        "source": "France Travail",
        "url": df.get("origineOffre.urlOrigine"),
    })


# ==================================================
# MERGE SOURCES
# ==================================================
def merge_sources(
    df_wttj: pd.DataFrame | None = None,
    df_ft: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Merge WTTJ and France Travail into a common schema
    """

    frames = []

    if df_wttj is not None and not df_wttj.empty:
        frames.append(preprocess_wttj(df_wttj))

    if df_ft is not None and not df_ft.empty:
        frames.append(preprocess_ft(df_ft))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Basic safety cleaning
    for col in [
        "title", "description", "company", "location",
        "industry", "experience", "education", "url"
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    return df
