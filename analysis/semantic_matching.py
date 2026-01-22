from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# ==================================================
# LOAD MODEL (ONCE)
# ==================================================
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)


# ==================================================
# HELPERS
# ==================================================
def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts"""
    return model.encode(texts, normalize_embeddings=True)


def semantic_overlap(
    cv_skills: list[str],
    job_skills: list[str],
    threshold: float = 0.75
) -> dict:
    """
    Compute semantic overlap between CV skills and job skills
    """

    if not cv_skills or not job_skills:
        return {
            "matched": [],
            "missing": job_skills,
            "coverage": 0.0
        }

    cv_emb = embed(cv_skills)
    job_emb = embed(job_skills)

    sim_matrix = cosine_similarity(job_emb, cv_emb)

    matched = []
    missing = []

    for i, job_skill in enumerate(job_skills):
        max_sim = sim_matrix[i].max()
        if max_sim >= threshold:
            matched.append(job_skill)
        else:
            missing.append(job_skill)

    coverage = len(matched) / len(job_skills)

    return {
        "matched": matched,
        "missing": missing,
        "coverage": round(coverage, 2)
    }
