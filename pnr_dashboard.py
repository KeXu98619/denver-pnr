"""
Denver PnR Trip Distribution Dashboard
---------------------------------------
Streamlit app visualising:
  1. Spatial trip distribution on census-tract level (choropleth + PnR geofence)
  2. Time-of-day bar chart for the selected station

Run with:
    streamlit run pnr_dashboard.py
"""

import os
import base64
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium

# ── Password gate ──────────────────────────────────────────────────────────────

def require_password():
    def _check():
        if st.session_state.get("pw_input", "") == st.secrets["APP_PASSWORD"]:
            st.session_state["authed"] = True
            st.session_state.pop("pw_input", None)
        else:
            st.session_state["authed"] = False

    if "authed" not in st.session_state or not st.session_state["authed"]:
        st.text_input("Password", type="password", key="pw_input", on_change=_check)
        if "authed" in st.session_state and st.session_state["authed"] is False:
            st.error("Incorrect password.")
        st.stop()

# ── Configuration ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Denver PnR Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

require_password()

st.markdown("""
<style>
[data-testid="stSidebar"] { position: relative; }
#sidebar-brand {
    position: fixed; left: 14px; bottom: 12px; z-index: 10;
    background: transparent; border: none; padding: 0;
}
#sidebar-brand img { height: 70px; display: block; }
</style>
""", unsafe_allow_html=True)

_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA       = os.path.join(_DIR, "Data")
TRACTS_SHP = os.path.join(DATA, "tl_2022_08_tract.shp")

# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_data
def load_trips():
    p2t = pd.read_csv(os.path.join(DATA, "pnr_to_tract_trips.csv"))
    t2p = pd.read_csv(os.path.join(DATA, "tract_to_pnr_trips.csv"))
    p2t["tract_id"] = p2t["tract_id"].astype(str).str.zfill(11)
    t2p["tract_id"] = t2p["tract_id"].astype(str).str.zfill(11)
    return p2t, t2p


@st.cache_data
def load_tod():
    xl  = os.path.join(DATA, "Denver PNR 2024 results_cleaned.xlsx")
    hro = pd.read_excel(xl, sheet_name="tod_o")
    hrd = pd.read_excel(xl, sheet_name="tod_d")
    return hro, hrd


@st.cache_data
def load_geo():
    # PnR geofences – original GeoJSON is already WGS84
    region  = gpd.read_file(os.path.join(DATA, "Denver PNR 2024 weekdays.geojson"))
    pnr_gdf = region[~region["name"].str.startswith("Region")].copy()
    if pnr_gdf.crs is None or pnr_gdf.crs.to_epsg() != 4326:
        pnr_gdf = pnr_gdf.set_crs(4326, allow_override=True)

    # Census tracts (Colorado)
    tracts = gpd.read_file(TRACTS_SHP)[["GEOID", "geometry"]].copy()
    if tracts.crs.to_epsg() != 4326:
        tracts = tracts.to_crs(4326)
    tracts["GEOID"] = tracts["GEOID"].astype(str)

    return pnr_gdf, tracts


# ── Helpers ────────────────────────────────────────────────────────────────────

def find_station_col(df: pd.DataFrame, candidates: set) -> str:
    """Return the column whose values best overlap with `candidates`."""
    best_col, best_n = df.columns[0], 0
    for col in df.select_dtypes(include="object").columns:
        n = df[col].isin(candidates).sum()
        if n > best_n:
            best_n, best_col = n, col
    return best_col


def hr_cols(df: pd.DataFrame) -> list:
    """Return columns whose names start with 'hr', preserving DataFrame order."""
    return [c for c in df.columns if str(c).lower().startswith("hr")]


def hr_label(col: str) -> str:
    """
    Convert column name to a readable time label.
      'hr6'   → '6am – 7am'
      'hr0-6' → '12am – 6am'
    """
    raw = str(col).lower().replace("hr", "").strip()

    def _fmt(h: int) -> str:
        h = int(h) % 24
        sfx = "am" if h < 12 else "pm"
        return f"{h % 12 or 12}{sfx}"

    if "-" in raw:
        a, b = raw.split("-")
        return f"{_fmt(a)} – {_fmt(b)}"
    if "," in raw:
        a, b = raw.split(",")
        return f"{_fmt(a)} – {_fmt(b)}"
    h = int(raw)
    return f"{_fmt(h)} – {_fmt(h + 1)}"


def polygon_outline(geom) -> tuple:
    """Return (lats, lons) for the outer ring of a (Multi)Polygon in WGS84."""
    if geom is None:
        return [], []
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    coords = list(geom.exterior.coords)
    lons = [c[0] for c in coords] + [coords[0][0]]   # close ring
    lats = [c[1] for c in coords] + [coords[0][1]]
    return lats, lons


# ── Load data ──────────────────────────────────────────────────────────────────

with st.spinner("Loading data…"):
    pnr_to_tract, tract_to_pnr = load_trips()
    hro, hrd                    = load_tod()
    pnr_gdf, tracts_gdf         = load_geo()

all_stations = sorted(
    set(pnr_to_tract["pnr"].unique()) | set(tract_to_pnr["pnr"].unique())
)

# ── Sidebar controls ───────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Filters")
    selected  = st.selectbox("PnR Station", all_stations)
    direction = st.radio(
        "Direction",
        ["Outgoing  (PnR → Tracts)", "Incoming  (Tracts → PnR)"],
    )

    # Logo pinned to the bottom of the sidebar
    _logo_path = os.path.join(_DIR, "logo.webp")
    if os.path.exists(_logo_path):
        _b64 = base64.b64encode(open(_logo_path, "rb").read()).decode("ascii")
        st.markdown(
            f"<div id='sidebar-brand'><img src='data:image/webp;base64,{_b64}'/></div>",
            unsafe_allow_html=True,
        )

is_out = direction.startswith("Outgoing")

# ── Filter trip data for selected station + direction ──────────────────────────

if is_out:
    trip_df = pnr_to_tract[pnr_to_tract["pnr"] == selected][["tract_id", "trips"]].copy()
else:
    trip_df = tract_to_pnr[tract_to_pnr["pnr"] == selected][["tract_id", "trips"]].copy()

total_trips = trip_df["trips"].sum()
trip_df["share_pct"] = (
    (trip_df["trips"] / total_trips * 100).round(2) if total_trips > 0 else 0.0
)

# ── Page header ────────────────────────────────────────────────────────────────

dir_str = "Outgoing (PnR → Tracts)" if is_out else "Incoming (Tracts → PnR)"
st.title(selected)
st.caption(f"Direction: **{dir_str}** · Total trips: **{total_trips:,.0f}**")

col_map, col_chart = st.columns([3, 2])

# ══════════════════════════════════════════════════════════════════════════════
# LEFT  –  SPATIAL MAP
# ══════════════════════════════════════════════════════════════════════════════

with col_map:
    st.subheader("Spatial Distribution by Census Tract")

    # Merge trip metrics onto tract geometries (inner → only tracts with trips)
    map_gdf = tracts_gdf.merge(trip_df, left_on="GEOID", right_on="tract_id", how="inner")

    if map_gdf.empty:
        st.info("No trip data found for this station / direction combination.")
    else:
        center_lat = float(map_gdf.geometry.centroid.y.mean())
        center_lon = float(map_gdf.geometry.centroid.x.mean())

        map_gdf = map_gdf.copy()
        map_gdf["trips_fmt"] = map_gdf["trips"].apply(lambda x: f"{x:,.0f}")
        map_gdf["share_fmt"] = map_gdf["share_pct"].apply(lambda x: f"{x:.2f}%")

        geo_data = map_gdf[["GEOID", "geometry"]].to_json()
        tip_data = map_gdf[["GEOID", "tract_id", "trips_fmt", "share_fmt", "geometry"]].to_json()

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=9,
            tiles="OpenStreetMap",
        )

        # ── Choropleth: tracts coloured by trip count ──────────────────────────
        folium.Choropleth(
            geo_data=geo_data,
            name="Trips",
            data=map_gdf[["GEOID", "trips"]],
            columns=["GEOID", "trips"],
            key_on="feature.properties.GEOID",
            fill_color="Blues",
            fill_opacity=0.75,
            line_opacity=0.4,
            line_weight=0.4,
            legend_name="Trips",
            nan_fill_color="white",
        ).add_to(m)

        # Transparent overlay so tooltips show trip values on hover
        folium.GeoJson(
            tip_data,
            style_function=lambda x: {"fillOpacity": 0, "weight": 0},
            tooltip=folium.GeoJsonTooltip(
                fields=["tract_id", "trips_fmt", "share_fmt"],
                aliases=["Tract ID:", "Trips:", "Share:"],
                style="font-size: 11px;",
            ),
        ).add_to(m)

        # ── PnR geofence outline ───────────────────────────────────────────────
        pnr_row = pnr_gdf[pnr_gdf["name"] == selected]
        if not pnr_row.empty:
            folium.GeoJson(
                pnr_row.to_json(),
                style_function=lambda x: {
                    "color": "#e63946",
                    "weight": 3,
                    "fillOpacity": 0,
                },
                tooltip=selected,
                name=selected,
            ).add_to(m)

        st_folium(m, use_container_width=True, height=520, returned_objects=[])

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT  –  TIME-OF-DAY CHART
# ══════════════════════════════════════════════════════════════════════════════

with col_chart:
    # tod_o  = trips originating from PnR (outgoing)
    # tod_d  = trips arriving at PnR       (incoming)
    tod_src     = hro if is_out else hrd
    station_col = find_station_col(tod_src, set(all_stations))
    tod_row     = tod_src[tod_src[station_col] == selected]

    if tod_row.empty:
        st.info(f"No time-of-day data for **{selected}**.")
    else:
        hr_columns = hr_cols(tod_src)
        tod_values = tod_row[hr_columns].iloc[0].values.astype(float)
        tod_labels = [hr_label(c) for c in hr_columns]

        fig_tod = go.Figure(go.Bar(
            x=tod_labels,
            y=tod_values,
            marker_color="#1d6ea6",
            hovertemplate="<b>%{x}</b><br>Trips: %{y:,.0f}<extra></extra>",
        ))
        fig_tod.update_layout(
            xaxis_title="Hour",
            yaxis_title="Trips",
            xaxis_tickangle=-40,
            margin=dict(l=0, r=10, t=10, b=0),
            height=340,
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis=dict(gridcolor="#e8e8e8", zeroline=False),
            bargap=0.25,
        )

        st.subheader("Time-of-Day Profile")
        st.plotly_chart(fig_tod, use_container_width=True)

        src_label = "originating from" if is_out else "arriving at"
        st.caption(f"Trips {src_label} **{selected}** by time of day.")

