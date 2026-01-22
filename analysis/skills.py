import pandas as pd
from collections import Counter


# ==================================================
# HELPERS
# ==================================================
def flatten(list_of_lists):
    """Flatten list of lists safely"""
    if not isinstance(list_of_lists, list):
        return []
    return [
        item
        for sublist in list_of_lists
        if isinstance(sublist, list)
        for item in sublist
        if item not in (None, "", " ")
    ]


def _safe_pct(count: int, total: int, ndigits: int = 2) -> float:
    """Safe percentage computation"""
    if total <= 0:
        return 0.0
    return round(100 * count / total, ndigits)


# ==================================================
# ======== FRANCE TRAVAIL ANALYSIS =================
# ==================================================

def analyze_hard_skills_ft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze hard skills for France Travail
    (required / optional respected)
    """

    total_jobs = len(df)
    if total_jobs == 0:
        return pd.DataFrame(
            columns=[
                "skill",
                "required_count",
                "optional_count",
                "required_pct",
                "optional_pct",
            ]
        )

    required_skills = flatten(df.get("skills_hard_required", []).tolist())
    optional_skills = flatten(df.get("skills_hard_optional", []).tolist())

    req_counter = Counter(required_skills)
    opt_counter = Counter(optional_skills)

    records = []
    all_skills = set(req_counter.keys()) | set(opt_counter.keys())

    for skill in all_skills:
        req_count = req_counter.get(skill, 0)
        opt_count = opt_counter.get(skill, 0)

        records.append({
            "skill": skill,
            "required_count": req_count,
            "optional_count": opt_count,
            "required_pct": _safe_pct(req_count, total_jobs),
            "optional_pct": _safe_pct(opt_count, total_jobs),
        })

    if not records:
        return pd.DataFrame(
            columns=[
                "skill",
                "required_count",
                "optional_count",
                "required_pct",
                "optional_pct",
            ]
        )

    return (
        pd.DataFrame(records)
        .sort_values(["required_pct", "optional_pct"], ascending=False)
        .reset_index(drop=True)
    )


def analyze_soft_skills_ft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze soft skills for France Travail
    (no required / optional)
    """

    total_jobs = len(df)
    if total_jobs == 0:
        return pd.DataFrame(columns=["skill", "count", "pct"])

    soft_skills = flatten(df.get("skills_soft", []).tolist())
    counter = Counter(soft_skills)

    if not counter:
        return pd.DataFrame(columns=["skill", "count", "pct"])

    data = [
        {
            "skill": skill,
            "count": count,
            "pct": _safe_pct(count, total_jobs),
        }
        for skill, count in counter.items()
    ]

    return (
        pd.DataFrame(data)
        .sort_values("pct", ascending=False)
        .reset_index(drop=True)
    )


def detect_emerging_skills_ft(
    df_hard_skills: pd.DataFrame,
    min_pct: float = 5,
    max_pct: float = 15
) -> pd.DataFrame:
    """
    Detect emerging hard skills for France Travail
    based on REQUIRED skills only
    """

    if df_hard_skills.empty:
        return df_hard_skills.copy()

    return (
        df_hard_skills[
            (df_hard_skills["required_pct"] >= min_pct) &
            (df_hard_skills["required_pct"] <= max_pct)
        ]
        .sort_values("required_pct", ascending=False)
        .reset_index(drop=True)
    )


def analyze_skills_ft(df_jobs: pd.DataFrame) -> dict:
    """
    Complete skill analysis pipeline for France Travail
    """

    hard_skills = analyze_hard_skills_ft(df_jobs)
    soft_skills = analyze_soft_skills_ft(df_jobs)
    emerging_skills = detect_emerging_skills_ft(hard_skills)

    return {
        "hard_skills": hard_skills,
        "soft_skills": soft_skills,
        "emerging_skills": emerging_skills,
    }


# ==================================================
# ===== WELCOME TO THE JUNGLE ANALYSIS ==============
# ==================================================

def analyze_skills_wttj(df: pd.DataFrame) -> dict:
    """
    Skill analysis for Welcome to the Jungle
    (no required / optional, frequency-based)
    """

    total_jobs = len(df)

    if total_jobs == 0:
        empty = pd.DataFrame(columns=["skill", "count", "pct"])
        return {
            "hard_skills": empty,
            "soft_skills": empty,
        }

    # -------- Hard skills
    hard_skills = flatten(df.get("skills_hard_required", []).tolist())
    hard_counter = Counter(hard_skills)

    if hard_counter:
        hard_df = pd.DataFrame([
            {
                "skill": skill,
                "count": count,
                "pct": _safe_pct(count, total_jobs),
            }
            for skill, count in hard_counter.items()
        ])
        hard_df = hard_df.sort_values("pct", ascending=False).reset_index(drop=True)
    else:
        hard_df = pd.DataFrame(columns=["skill", "count", "pct"])

    # -------- Soft skills
    soft_skills = flatten(df.get("skills_soft", []).tolist())
    soft_counter = Counter(soft_skills)

    if soft_counter:
        soft_df = pd.DataFrame([
            {
                "skill": skill,
                "count": count,
                "pct": _safe_pct(count, total_jobs),
            }
            for skill, count in soft_counter.items()
        ])
        soft_df = soft_df.sort_values("pct", ascending=False).reset_index(drop=True)
    else:
        soft_df = pd.DataFrame(columns=["skill", "count", "pct"])

    return {
        "hard_skills": hard_df,
        "soft_skills": soft_df,
    }
