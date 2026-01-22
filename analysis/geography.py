import pandas as pd
import folium
from folium.plugins import MarkerCluster


# ==================================================
# FRANCE TRAVAIL — MAP ANALYSIS
# ==================================================
def create_ft_jobs_map(
    df: pd.DataFrame,
    center: list[float] = [46.5, 2],
    zoom_start: int = 5
):
    """
    Create an interactive map showing France Travail job offers
    (only jobs with reliable latitude / longitude)
    """

    # Safety check
    required_cols = {"latitude", "longitude", "title", "location"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            "DataFrame must contain latitude, longitude, title, location columns"
        )

    # Base map
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles="CartoDB positron"
    )

    marker_cluster = MarkerCluster().add_to(m)

    df_geo = df.dropna(subset=["latitude", "longitude"])

    for _, row in df_geo.iterrows():
        popup = (
            f"<b>{row.get('title','')}</b><br>"
            f"{row.get('company','')}<br>"
            f"{row.get('location','')}"
        )

        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=popup,
            icon=folium.Icon(
                color="darkblue",
                icon="briefcase",
                prefix="fa"
            )
        ).add_to(marker_cluster)

    return m


# ==================================================
# BASIC GEO STATS (OPTIONAL KPI)
# ==================================================
def jobs_by_location(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Count number of job offers by location
    """
    return (
        df["location"]
        .value_counts()
        .head(top_n)
        .reset_index(name="count")
        .rename(columns={"index": "location"})
    )

