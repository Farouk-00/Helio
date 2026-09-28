"""Cartes plotly partagées (fonds MapLibre, sans jeton)."""
from __future__ import annotations

import base64
import io

import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from plotly.colors import sample_colorscale
from shapely.geometry import MultiPolygon, Point, Polygon

from core.config import LCZ_TILES_MAXZOOM, LCZ_TILES_URL


def outline(geom) -> tuple[list, list]:
    """Contours extérieurs d'un (multi)polygone, séparés par None pour plotly."""
    lons, lats = [], []
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom] if isinstance(geom, Polygon) else []
    for p in polys:
        x, y = p.exterior.xy
        lons += list(x) + [None]
        lats += list(y) + [None]
    return lons, lats


def disk_wgs84(lat: float, lon: float, radius: float):
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    return pt.to_crs(pt.estimate_utm_crs()).buffer(radius).to_crs("EPSG:4326").iloc[0]


def disk_trace(lat: float, lon: float, radius: float, name: str | None = None) -> go.Scattermap:
    lons, lats = outline(disk_wgs84(lat, lon, radius))
    label = name or (f"Rayon {radius / 1000:g} km" if radius >= 1000 else f"Rayon {radius:g} m")
    return go.Scattermap(lon=lons, lat=lats, mode="lines", line=dict(width=2, color="#1f4e8c"),
                         name=label, hoverinfo="skip")


def site_marker(site: dict) -> go.Scattermap:
    return go.Scattermap(lat=[site["lat"]], lon=[site["lon"]], mode="markers",
                         marker=dict(size=13, color="#1f4e8c"), name="Site",
                         hovertext=[site["label"]], hoverinfo="text")


def raster_layer(img: np.ndarray, transform, crs, vmin: float, vmax: float,
                 colorscale: str = "Inferno", opacity: float = 0.75) -> dict:
    """Image (NaN transparents) posée sur la carte par ses 4 coins, en couche MapLibre."""
    lut = np.array([[int(c) for c in s[s.index("(") + 1:-1].split(",")]
                    for s in sample_colorscale(colorscale, np.linspace(0, 1, 256), colortype="rgb")],
                   dtype=np.uint8)
    idx = np.clip((img - vmin) / max(vmax - vmin, 1e-6) * 255, 0, 255)
    rgba = np.zeros(img.shape + (4,), dtype=np.uint8)
    ok = np.isfinite(img)
    rgba[ok, :3] = lut[idx[ok].astype(int)]
    rgba[ok, 3] = 255
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    h, w = img.shape
    corners = [transform * (0, 0), transform * (w, 0), transform * (w, h), transform * (0, h)]
    ll = gpd.GeoSeries([Point(x, y) for x, y in corners], crs=crs).to_crs("EPSG:4326")
    return dict(sourcetype="image", source="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                coordinates=[[p.x, p.y] for p in ll], below="traces", opacity=opacity)


def colorbar_trace(lat: float, lon: float, vmin: float, vmax: float, title: str,
                   colorscale: str = "Inferno") -> go.Scattermap:
    """Trace invisible qui ne sert qu'à afficher l'échelle de couleur d'une image raster."""
    return go.Scattermap(lat=[lat, lat], lon=[lon, lon], mode="markers", hoverinfo="skip",
                         showlegend=False,
                         marker=dict(size=0, color=[vmin, vmax], colorscale=colorscale, cmin=vmin,
                                     cmax=vmax, showscale=True,
                                     colorbar=dict(title=title, thickness=12, len=0.6, x=0.99)))


def layout_map(fig: go.Figure, lat: float, lon: float, zoom: float, height: int = 520,
               lcz_tiles: bool = False, layers: list | None = None) -> go.Figure:
    layers = list(layers or [])
    if lcz_tiles:  # tuiles LCZ nationales du Cerema (servies jusqu'au zoom 14)
        layers.append(dict(sourcetype="raster", source=[LCZ_TILES_URL], below="traces",
                           opacity=0.65, maxzoom=LCZ_TILES_MAXZOOM + 1))
    fig.update_layout(map=dict(style="carto-positron", center=dict(lat=lat, lon=lon), zoom=zoom,
                               layers=layers),
                      height=height, margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                                  bgcolor="rgba(255,255,255,0.8)"))
    return fig
