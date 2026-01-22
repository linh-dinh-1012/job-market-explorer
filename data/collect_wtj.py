from __future__ import annotations

import time
import re
import json
import urllib.parse
from typing import Callable, Optional, Any, Dict, List

import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common import TimeoutException


# ==================================================
# DRIVER
# ==================================================
def create_driver(headless: bool = True):
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)


# ==================================================
# URL helper: set/override ?query=... safely
# ==================================================
def _build_url_with_query(base_url: str, keyword: str) -> str:
    """
    Ensure base_url has query=<keyword> (override if already exists).
    Works even if base_url originally has no 'query=' param.
    """
    parts = urllib.parse.urlsplit(base_url)
    qs = urllib.parse.parse_qs(parts.query, keep_blank_values=True)

    qs["query"] = [keyword]  # do NOT pre-encode; urlencode will handle it
    # Reset page if present (optional but cleaner)
    if "page" in qs:
        qs["page"] = ["1"]

    new_query = urllib.parse.urlencode(qs, doseq=True, safe="[]")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))


# ==================================================
# NORMALIZERS / FILTER HELPERS
# ==================================================
def _normalize_contract(val: str) -> str:
    s = (val or "").upper().strip()
    if "INTERN" in s or "INTERNSHIP" in s or "STAGE" in s:
        return "INTERN"
    if "TEMP" in s or "TEMPORARY" in s or "CDD" in s or "CONTRACT" in s:
        return "TEMPORAIN"
    if "FULL" in s or "CDI" in s:
        return "FULL_TIME"
    return "OTHER"


def _extract_experience_years(meta: dict) -> Optional[int]:
    exp = meta.get("experienceRequirements")
    if isinstance(exp, dict):
        months = exp.get("monthsOfExperience")
        if months is not None:
            try:
                return int(round(float(months) / 12))
            except Exception:
                pass

    for key in ["experienceRequirements", "description"]:
        v = meta.get(key)
        if isinstance(v, str):
            m = re.search(r"(\d+)\s*(ans|an)\b", v.lower())
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass
    return None


def _location_match(location: str, filters: Optional[List[str]]) -> bool:
    if not filters:
        return True
    loc = (location or "").lower()
    return any(f.lower().strip() in loc for f in filters if f.strip())


def _contract_match(contract_norm: str, allowed: Optional[List[str]]) -> bool:
    if not allowed:
        return True
    allowed_set = {a.upper().strip() for a in allowed}
    return contract_norm.upper() in allowed_set


def _experience_match(years: Optional[int], min_years: Optional[int], max_years: Optional[int]) -> bool:
    if years is None:
        if min_years is not None or max_years is not None:
            return False
        return True
    if min_years is not None and years < min_years:
        return False
    if max_years is not None and years > max_years:
        return False
    return True


# ==================================================
# MAIN COLLECTOR
# ==================================================
def fetch_wttj_offers(
    keywords: List[str],
    base_url: str,
    max_pages: int = 5,
    headless: bool = True,
    sleep_time: float = 1.0,
    # filters (post detail)
    min_experience_years: Optional[int] = None,
    max_experience_years: Optional[int] = None,
    location_filters: Optional[List[str]] = None,
    contract_filters: Optional[List[str]] = None,
    # progress
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> pd.DataFrame:
    """
    Collect job offers from Welcome to the Jungle (Selenium + JSON-LD)
    + filters applied AFTER detail extraction
    + progress callback
    """

    urls = [_build_url_with_query(base_url, k) for k in keywords]

    driver = create_driver(headless=headless)
    all_jobs: List[Dict[str, Any]] = []

    try:
        # --------------------------
        # LISTING PAGES
        # --------------------------
        total_listing_steps = max(1, len(urls) * max_pages)
        listing_done = 0

        for kw, url in zip(keywords, urls):
            driver.get(url)
            page = 1

            while page <= max_pages:
                listing_done += 1
                if on_progress:
                    on_progress({
                        "stage": "listing",
                        "keyword": kw,
                        "page": page,
                        "max_pages": max_pages,
                        "progress": min(0.20, 0.20 * (listing_done / total_listing_steps)),
                        "msg": f"Listing '{kw}' — page {page}/{max_pages}",
                    })

                try:
                    cards = WebDriverWait(driver, 6).until(
                        EC.presence_of_all_elements_located(
                            (By.XPATH, "//li[@data-testid='search-results-list-item-wrapper']")
                        )
                    )
                except TimeoutException:
                    break

                for card in cards:
                    try:
                        link_elem = card.find_element(By.XPATH, ".//a[h2]")
                        all_jobs.append({
                            "title": link_elem.text.strip(),
                            "link": link_elem.get_attribute("href"),
                            "company": card.find_element(
                                By.XPATH, ".//span[contains(@class, 'wui-text')]"
                            ).text.strip(),
                            "date": card.find_element(By.TAG_NAME, "time").get_attribute("datetime")
                        })
                    except Exception:
                        continue

                # Pagination
                try:
                    next_btn = WebDriverWait(driver, 2).until(
                        EC.element_to_be_clickable(
                            (By.XPATH, '//nav[@aria-label="Pagination"]//li[last()]//a')
                        )
                    )
                    if next_btn.get_attribute("aria-disabled") == "false":
                        next_btn.click()
                        page += 1
                        time.sleep(sleep_time)
                    else:
                        break
                except TimeoutException:
                    break

        # Deduplicate by link
        unique_jobs = list({j["link"]: j for j in all_jobs}.values())

        # --------------------------
        # DETAIL PAGES (JSON-LD)
        # --------------------------
        data: List[Dict[str, Any]] = []
        total = len(unique_jobs)

        for i, job in enumerate(unique_jobs, start=1):
            if on_progress:
                on_progress({
                    "stage": "detail",
                    "i": i,
                    "total": total,
                    "progress": 0.20 + (0.80 * (i / max(1, total))),
                    "msg": f"Detail {i}/{total}: {job.get('title','')[:60]}",
                })

            driver.get(job["link"])

            # Expand description
            try:
                voir_plus = WebDriverWait(driver, 3).until(
                    EC.element_to_be_clickable((By.XPATH, "//span[contains(text(),'Voir plus')]"))
                )
                driver.execute_script("arguments[0].scrollIntoView({block:'center'});", voir_plus)
                voir_plus.click()
            except Exception:
                pass

            row = {
                **job,
                "contract": "",
                "contract_norm": "",
                "location": "",
                "salary": "",
                "experience": "",
                "experience_years": None,
                "education": "",
                "industry": "",
                "company_info": "",
                "description": "",
                "source": "Welcome to the Jungle",
            }

            try:
                script = driver.find_element(By.XPATH, "//script[@type='application/ld+json']")
                meta = json.loads(script.get_attribute("innerHTML"))

                row["description"] = meta.get("description", "")
                row["contract"] = meta.get("employmentType", "") or ""
                row["contract_norm"] = _normalize_contract(str(row["contract"]))
                row["industry"] = meta.get("industry", "") or ""

                if "baseSalary" in meta:
                    val = meta["baseSalary"]["value"]
                    if isinstance(val, dict):
                        row["salary"] = (
                            f"{val.get('minValue','')}-{val.get('maxValue','')} "
                            f"{meta['baseSalary'].get('currency','')}/"
                            f"{val.get('unitText','').lower()}"
                        )

                if "jobLocation" in meta:
                    addr = meta["jobLocation"][0]["address"]
                    row["location"] = f"{addr.get('addressLocality','')}, {addr.get('postalCode','')}"

                exp_years = _extract_experience_years(meta)
                row["experience_years"] = exp_years
                if exp_years is not None:
                    row["experience"] = f"{exp_years} ans d’expérience"

                if "educationRequirements" in meta:
                    row["education"] = meta["educationRequirements"].get("credentialCategory", "") or ""

                if "hiringOrganization" in meta:
                    org = meta["hiringOrganization"]
                    row["company_info"] = (
                        f"{org.get('name','')} | "
                        f"{org.get('address',{}).get('streetAddress','')} | "
                        f"{org.get('sameAs','')}"
                    )

            except Exception:
                pass

            data.append(row)

        df = pd.DataFrame(data)

        # --------------------------
        # APPLY FILTERS (optional)
        # --------------------------
        if not df.empty:
            if "contract_norm" not in df.columns:
                df["contract_norm"] = df.get("contract", "").astype(str).apply(_normalize_contract)

            mask = []
            for _, r in df.iterrows():
                ok = True
                ok = ok and _location_match(r.get("location", ""), location_filters)
                ok = ok and _contract_match(r.get("contract_norm", ""), contract_filters)
                ok = ok and _experience_match(
                    r.get("experience_years", None),
                    min_experience_years,
                    max_experience_years,
                )
                mask.append(ok)

            df = df[pd.Series(mask, index=df.index)].reset_index(drop=True)

        if on_progress:
            on_progress({"stage": "done", "progress": 1.0, "msg": f"Done ({len(df)} offers)"})

        return df

    finally:
        try:
            driver.quit()
        except Exception:
            pass
