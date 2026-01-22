from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import re

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ==================================================
# MODEL (loaded once)
# ==================================================
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_MODEL: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


# ==================================================
# TEXT / LIST NORMALIZATION
# ==================================================
def norm_text(s: Any) -> str:
    """Lowercase + light cleanup (safe for None)."""
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def norm_list(x: Any) -> List[str]:
    """Ensure list[str], lowercase, trimmed, remove empties."""
    if not isinstance(x, list):
        return []
    out = []
    for v in x:
        if isinstance(v, str):
            vv = norm_text(v)
            if vv:
                out.append(vv)
    return out


def to_set(x: Any) -> set:
    return set(norm_list(x))


def coverage(cv_set: set, job_set: set) -> float:
    """Exact coverage ratio. If no job_set, return 1.0."""
    if not job_set:
        return 1.0
    return len(cv_set & job_set) / len(job_set)


# ==================================================
# EMBEDDINGS
# ==================================================
def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed list of short texts (skills/phrases)."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)  # fallback shape; not used if empty
    model = get_model()
    return model.encode(texts, normalize_embeddings=True)


def embed_text(text: str) -> np.ndarray:
    """Embed a single text string."""
    model = get_model()
    return model.encode([text], normalize_embeddings=True)


def semantic_overlap(
    cv_items: List[str],
    job_items: List[str],
    threshold: float = 0.75
) -> Dict[str, Any]:
    """
    Semantic overlap between two lists of short items (skills).
    Returns matched/missing and coverage.
    """
    cv_items = [norm_text(x) for x in cv_items if isinstance(x, str) and norm_text(x)]
    job_items = [norm_text(x) for x in job_items if isinstance(x, str) and norm_text(x)]

    if not job_items:
        return {"matched": [], "missing": [], "coverage": 1.0}

    if not cv_items:
        return {"matched": [], "missing": job_items, "coverage": 0.0}

    cv_emb = embed_texts(cv_items)
    job_emb = embed_texts(job_items)

    sim = cosine_similarity(job_emb, cv_emb)

    matched, missing = [], []
    for i, job_item in enumerate(job_items):
        if sim[i].max() >= threshold:
            matched.append(job_item)
        else:
            missing.append(job_item)

    cov = len(matched) / len(job_items) if job_items else 1.0
    return {"matched": matched, "missing": missing, "coverage": round(cov, 2)}


def semantic_similarity_text(
    cv_text: str,
    job_text: str
) -> float:
    """
    Semantic similarity between two longer texts (CV summary/experience vs job description).
    Returns cosine similarity in [0,1] (approx).
    """
    cv_text = norm_text(cv_text)
    job_text = norm_text(job_text)

    if not cv_text or not job_text:
        return 0.0

    a = embed_text(cv_text)  # (1, d)
    b = embed_text(job_text)  # (1, d)
    sim = float(cosine_similarity(a, b)[0, 0])
    # cosine similarity with normalized embeddings is [-1,1], but typically [0,1] for semantic space
    return round(sim, 3)


# ==================================================
# CORE MATCHING (skills + description)
# ==================================================
def match_cv_job(
    cv: Dict[str, Any],
    job: pd.Series,
    *,
    use_semantic_skills: bool = True,
    skill_threshold: float = 0.75,
    use_description_semantic: bool = True,
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Match one CV against one job offer.
    CV expected keys (all optional but recommended):
      - skills_hard: list[str]
      - skills_soft: list[str]
      - languages: list[str]
      - text: str   (CV summary/experience text for description matching)

    Job expected fields (from preprocessing V2):
      - skills_hard_required: list[str]
      - skills_hard_optional: list[str]
      - skills_soft: list[str]
      - languages_required: list[str]
      - languages_optional: list[str]
      - description: str
    """

    # Default weights: skills are constraints; description helps ranking
    if weights is None:
        weights = {
            "hard_required": 0.45,
            "hard_optional": 0.15,
            "soft": 0.10,
            "languages": 0.10,
            "description": 0.20,
        }

    # ---- CV inputs ----
    cv_hard_list = norm_list(cv.get("skills_hard"))
    cv_soft_list = norm_list(cv.get("skills_soft"))
    cv_lang_list = norm_list(cv.get("languages"))
    cv_text = norm_text(cv.get("text", ""))

    cv_hard = set(cv_hard_list)
    cv_soft = set(cv_soft_list)
    cv_lang = set(cv_lang_list)

    # ---- Job inputs ----
    job_req_list = norm_list(job.get("skills_hard_required"))
    job_opt_list = norm_list(job.get("skills_hard_optional"))
    job_soft_list = norm_list(job.get("skills_soft"))
    job_lang_req_list = norm_list(job.get("languages_required"))
    job_lang_opt_list = norm_list(job.get("languages_optional"))
    job_desc = norm_text(job.get("description", ""))

    job_req = set(job_req_list)
    job_opt = set(job_opt_list)
    job_soft = set(job_soft_list)
    job_lang_req = set(job_lang_req_list)
    job_lang_opt = set(job_lang_opt_list)

    # ---- Exact coverage ----
    hard_req_exact = coverage(cv_hard, job_req)
    hard_opt_exact = coverage(cv_hard, job_opt)
    soft_exact = coverage(cv_soft, job_soft)
    lang_req_exact = coverage(cv_lang, job_lang_req)

    # ---- Semantic fallback for skills (ONLY to reduce false negatives) ----
    hard_req = hard_req_exact
    hard_opt = hard_opt_exact
    lang_req = lang_req_exact

    sem_req_detail = None
    sem_opt_detail = None
    sem_lang_detail = None

    if use_semantic_skills:
        # required skills semantic (only if some missing)
        if job_req_list and (len(job_req - cv_hard) > 0):
            sem_req_detail = semantic_overlap(cv_hard_list, job_req_list, threshold=skill_threshold)
            hard_req = sem_req_detail["coverage"]

        # optional skills semantic
        if job_opt_list and (len(job_opt - cv_hard) > 0):
            sem_opt_detail = semantic_overlap(cv_hard_list, job_opt_list, threshold=skill_threshold)
            hard_opt = sem_opt_detail["coverage"]

        # required languages semantic (rarely needed but keeps logic consistent)
        if job_lang_req_list and (len(job_lang_req - cv_lang) > 0):
            sem_lang_detail = semantic_overlap(cv_lang_list, job_lang_req_list, threshold=skill_threshold)
            lang_req = sem_lang_detail["coverage"]

    # ---- Description semantic similarity ----
    desc_sim = 0.0
    if use_description_semantic:
        desc_sim = semantic_similarity_text(cv_text, job_desc)

    # ---- Score ----
    total = (
        weights["hard_required"] * hard_req
        + weights["hard_optional"] * hard_opt
        + weights["soft"] * soft_exact
        + weights["languages"] * lang_req
        + weights["description"] * desc_sim
    )

    # ---- Explainability: missing items (prefer semantic missing if computed) ----
    hard_required_missing = list((job_req - cv_hard))
    hard_optional_missing = list((job_opt - cv_hard))
    languages_required_missing = list((job_lang_req - cv_lang))
    languages_optional_missing = list((job_lang_opt - cv_lang))
    soft_missing = list((job_soft - cv_soft))

    if sem_req_detail is not None:
        hard_required_missing = sem_req_detail["missing"]
    if sem_opt_detail is not None:
        hard_optional_missing = sem_opt_detail["missing"]
    if sem_lang_detail is not None:
        languages_required_missing = sem_lang_detail["missing"]

    return {
        # Main score
        "score": round(total * 100, 1),

        # Subscores (useful for UI)
        "hard_required_coverage": round(hard_req, 2),
        "hard_optional_coverage": round(hard_opt, 2),
        "soft_coverage": round(soft_exact, 2),
        "languages_required_coverage": round(lang_req, 2),
        "description_similarity": round(desc_sim, 3),

        # Missing (explainability)
        "hard_required_missing": sorted(set(hard_required_missing)),
        "hard_optional_missing": sorted(set(hard_optional_missing)),
        "soft_skills_missing": sorted(set(soft_missing)),
        "languages_required_missing": sorted(set(languages_required_missing)),
        "languages_optional_missing": sorted(set(languages_optional_missing)),

        # Diagnostics
        "used_semantic_skills": bool(use_semantic_skills),
        "skill_threshold": skill_threshold,
        "used_description_semantic": bool(use_description_semantic),
        "weights": weights,
    }


# ==================================================
# MATCH CV AGAINST MARKET
# ==================================================
def match_cv_market(
    cv: Dict[str, Any],
    df_jobs: pd.DataFrame,
    *,
    min_score: float = 40.0,
    use_semantic_skills: bool = True,
    skill_threshold: float = 0.75,
    use_description_semantic: bool = True,
    weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Match a CV against all job offers (df_jobs) and return ranked results.
    """

    results: List[Dict[str, Any]] = []

    for _, job in df_jobs.iterrows():
        m = match_cv_job(
            cv,
            job,
            use_semantic_skills=use_semantic_skills,
            skill_threshold=skill_threshold,
            use_description_semantic=use_description_semantic,
            weights=weights,
        )

        if m["score"] >= min_score:
            results.append({
                "job_title": job.get("title", ""),
                "company": job.get("company", ""),
                "location": job.get("location", ""),
                "source": job.get("source", ""),
                "url": job.get("url", ""),
                **m,
            })

    if not results:
        return pd.DataFrame()

    return (
        pd.DataFrame(results)
        .sort_values(["score", "hard_required_coverage", "description_similarity"], ascending=False)
        .reset_index(drop=True)
    )


# ==================================================
# QUICK EXAMPLE 
# ==================================================
if __name__ == "__main__":
    # Example CV dict
    cv_example = {
        "skills_hard": ["python", "sql", "power bi", "data analysis"],
        "skills_soft": ["rigueur", "autonomie", "esprit d'analyse"],
        "languages": ["anglais"],
        "text": "Analyse de données, automatisation de reporting, dashboards, aide à la décision.",
    }
    pass
