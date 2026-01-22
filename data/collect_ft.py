import time
import requests
import pandas as pd
from data.ft_auth import get_ft_access_token


FT_SEARCH_URL = (
    "https://api.francetravail.io/partenaire/"
    "offresdemploi/v2/offres/search"
)

def fetch_all_ft_offers(
    token: str,
    keywords: str,
    location: str | None = None,
    contract_type: str | None = None,
    step: int = 150,
    max_results: int = 600
) -> pd.DataFrame:
    """
    Fetch job offers
    """

    headers = {"Authorization": f"Bearer {token}"}

    all_results = []
    start = 0

    while start < max_results:
        params = {
            "motsCles": keywords,
            "range": f"{start}-{start + step - 1}",
            "sort": "1"
        }

        if location:
            params["lieuTravail"] = location
        if contract_type:
            params["typeContrat"] = contract_type

        r = requests.get(FT_SEARCH_URL, headers=headers, params=params)
        r.raise_for_status()

        results = r.json().get("resultats", [])

        if not results:
            break

        all_results.extend(results)

        if len(results) < step:
            break

        start += step
        time.sleep(0.5)  

    return pd.DataFrame(all_results)


