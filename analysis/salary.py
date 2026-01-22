import pandas as pd
import re
from typing import Optional, Tuple


# ==================================================
# TEXT NORMALIZATION
# ==================================================
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ==================================================
# -------- FRANCE TRAVAIL HELPERS ------------------
# ==================================================
def extract_numbers(text: str) -> list:
    if not text:
        return []
    nums = re.findall(r"\d+(?:[\.,]\d+)?", text)
    return [float(n.replace(",", ".")) for n in nums]


def detect_unit(text: str) -> str:
    if "horaire" in text or "/h" in text:
        return "hourly"
    if "mois" in text or "mensuel" in text:
        return "monthly"
    return "annual"


def to_annual(amount: float, unit: str) -> float:
    if unit == "monthly":
        return amount * 12
    if unit == "hourly":
        return amount * 151.67 * 12
    return amount


def parse_salary_ft(text: str) -> Tuple[Optional[float], Optional[float]]:
    text = normalize_text(text)
    nums = extract_numbers(text)

    if not nums:
        return None, None

    unit = detect_unit(text)

    if len(nums) == 1:
        val = to_annual(nums[0], unit)
        return val, val

    return (
        to_annual(min(nums), unit),
        to_annual(max(nums), unit)
    )


# ==================================================
# -------- WTTJ HELPERS -----------------------------
# ==================================================
def has_salary_wttj(text: str) -> bool:
    """
    Detect if a salary is mentioned in WTTJ offer
    (no attempt to extract amount)
    """
    if not text:
        return False

    patterns = [
        "€", "eur", "euro",
        "k€", "k eur",
        "rémunération", "salaire",
        "package", "brut", "net"
    ]

    return any(p in text.lower() for p in patterns)


# ==================================================
# MAIN ANALYSIS
# ==================================================
def analyze_salary(df: pd.DataFrame, source: str) -> dict:
    """
    Analyze salary depending on source
    source ∈ {"France Travail", "Welcome to the Jungle"}
    """

    df = df.copy()
    total_offres = len(df)

    # ----------------- FRANCE TRAVAIL -----------------
    if source == "France Travail":
        df["salary_text"] = df["salary"].apply(normalize_text)

        df[["salary_min", "salary_max"]] = (
            df["salary_text"]
            .apply(lambda x: pd.Series(parse_salary_ft(x)))
        )

        df["has_salary"] = df["salary_min"].notna()

        with_salary = int(df["has_salary"].sum())

        salary_values = pd.concat(
            [df["salary_min"], df["salary_max"]]
        ).dropna()

        distribution = {}
        if not salary_values.empty:
            distribution = {
                "min": round(salary_values.min(), 0),
                "median": round(salary_values.median(), 0),
                "max": round(salary_values.max(), 0),
            }

        return {
            "source": source,
            "kpis": {
                "total_offres": total_offres,
                "offres_avec_salaire": with_salary,
                "pct_avec_salaire": round(100 * with_salary / total_offres, 1)
                if total_offres else 0,
            },
            "distribution": distribution,
            "df_salary": df,
        }

    # ----------------- WTTJ -----------------
    elif source == "Welcome to the Jungle":
        df["salary_text"] = df["salary"].apply(normalize_text)
        df["has_salary"] = df["salary_text"].apply(has_salary_wttj)

        with_salary = int(df["has_salary"].sum())

        return {
            "source": source,
            "kpis": {
                "total_offres": total_offres,
                "offres_avec_salaire": with_salary,
                "pct_avec_salaire": round(100 * with_salary / total_offres, 1)
                if total_offres else 0,
            },
            "distribution": None,
            "df_salary": df,
        }

    else:
        raise ValueError("source doit être 'France Travail' ou 'Welcome to the Jungle'")
