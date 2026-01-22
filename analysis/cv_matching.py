from typing import Dict, List
import pandas as pd


# ==================================================
# HELPERS
# ==================================================
def to_set(x) -> set:
    """Convert list to set safely"""
    if isinstance(x, list):
        return set([i.lower() for i in x])
    return set()


def coverage(cv_set: set, job_set: set) -> float:
    """Coverage ratio"""
    if not job_set:
        return 1.0
    return len(cv_set & job_set) / len(job_set)


# ==================================================
# CORE MATCHING LOGIC
# ==================================================
def match_cv_job(
    cv: Dict,
    job: pd.Series,
    weights: Dict[str, float] | None = None
) -> Dict:
    """
    Match one CV against one job offer
    """

    # Default weights (recruitment logic)
    if weights is None:
        weights = {
            "hard_required": 0.6,
            "hard_optional": 0.2,
            "soft": 0.1,
            "languages": 0.1,
        }

    # CV sets
    cv_hard = to_set(cv.get("skills_hard"))
    cv_soft = to_set(cv.get("skills_soft"))
    cv_lang = to_set(cv.get("languages"))

    # Job sets
    job_req = to_set(job.get("skills_hard_required"))
    job_opt = to_set(job.get("skills_hard_optional"))
    job_soft = to_set(job.get("skills_soft"))
    job_lang_req = to_set(job.get("languages_required"))

    # ---- Matching ----
    hard_required_score = coverage(cv_hard, job_req)
    hard_optional_score = coverage(cv_hard, job_opt)
    soft_score = coverage(cv_soft, job_soft)
    language_score = coverage(cv_lang, job_lang_req)

    # ---- Global score ----
    total_score = (
        weights["hard_required"] * hard_required_score
        + weights["hard_optional"] * hard_optional_score
        + weights["soft"] * soft_score
        + weights["languages"] * language_score
    )

    return {
        "score": round(total_score * 100, 1),

        # ---- Explainability ----
        "hard_required_missing": list(job_req - cv_hard),
        "hard_optional_missing": list(job_opt - cv_hard),
        "soft_skills_missing": list(job_soft - cv_soft),
        "languages_missing": list(job_lang_req - cv_lang),

        "hard_required_coverage": round(hard_required_score, 2),
        "hard_optional_coverage": round(hard_optional_score, 2),
        "soft_coverage": round(soft_score, 2),
        "languages_coverage": round(language_score, 2),
    }


# ==================================================
# MATCH CV AGAINST ALL JOBS
# ==================================================
def match_cv_market(
    cv: Dict,
    df_jobs: pd.DataFrame,
    min_score: float = 40
) -> pd.DataFrame:
    """
    Match CV against all job offers
    """

    results = []

    for _, job in df_jobs.iterrows():
        match = match_cv_job(cv, job)

        if match["score"] >= min_score:
            results.append({
                "job_title": job.get("title"),
                "company": job.get("company"),
                "location": job.get("location"),
                "source": job.get("source"),
                "url": job.get("url"),
                **match,
            })

    return (
        pd.DataFrame(results)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
