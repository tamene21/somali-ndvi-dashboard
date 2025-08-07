import streamlit as st
import rasterio
from rasterio.mask import mask 
import geopandas as gpd
import numpy as np
import pandas as pd
import leafmap.foliumap as leafmap
from rasterstats import zonal_stats
from localtileserver import TileClient, get_folium_tile_layer
import folium

st.set_page_config(layout="wide")
st.title("🌿 NDVI Dashboard – Somali Region (Ethiopia)")

# --- File paths ---
ndvi_tif = "somali_ndvi_2023.tif"
shapefile = "somali_woredas.geojson"  # You can replace with .shp if needed

# --- Load shapefile ---
st.sidebar.header("Region Boundaries")
gdf = gpd.read_file(shapefile)
gdf = gdf.to_crs("EPSG:4326")

# --- Load raster and mask values outside NDVI range ---
with rasterio.open(ndvi_tif) as src:
    ndvi_data = src.read(1)
    ndvi_data[(ndvi_data < -0.0645) | (ndvi_data > 0.7321)] = np.nan
    affine = src.transform

# --- Crop raster to Somali region boundary ---
with rasterio.open(ndvi_tif) as src:
    geoms = gdf.geometry.values
    geoms = [geom.__geo_interface__ for geom in geoms]
    out_image, out_transform = mask(src, geoms, crop=True, nodata=np.nan)
    masked_tif = "masked_ndvi.tif"
    meta = src.meta.copy()
    meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "nodata": np.nan
    })
    with rasterio.open(masked_tif, "w", **meta) as dst:
        dst.write(out_image)

# --- Compute zonal statistics ---
st.sidebar.header("Compute Zonal Stats")
st.sidebar.write("Calculating mean NDVI for each district...")

zs = zonal_stats(
    vectors=gdf,
    raster=masked_tif,
    stats=["mean"],
    geojson_out=True,
    nodata=np.nan
)

# Convert result to GeoDataFrame
zs_gdf = gpd.GeoDataFrame.from_features(zs)
zs_gdf["mean"] = zs_gdf["mean"].round(3)

# --- NDVI line graph in sidebar ---
st.sidebar.subheader("NDVI by Woreda (Line Graph)")
if "admin3Name" in zs_gdf.columns:
    chart_df = zs_gdf[["admin3Name", "mean"]].sort_values("mean")
    st.sidebar.line_chart(chart_df.set_index("admin3Name"))
else:
    st.sidebar.write("Column 'admin3Name' not found in data.")

# --- Display interactive map ---
st.subheader("🗺️ Interactive NDVI Map with District Boundaries")

m = leafmap.Map(center=[6.5, 45.5], zoom=6, basemap="CartoDB.Positron")
m.add_raster(masked_tif, layer_name="NDVI", colormap="RdYlGn", opacity=1.0, nodata=np.nan, nodata_color="rgba(0,0,0,0)")
m.add_gdf(zs_gdf, layer_name="Woreda NDVI Stats", info_mode="on_hover", style={"color": "blue", "weight": 1, "fillOpacity": 0.1})
m.to_streamlit(height=600, width=1200)

# --- Show summary table ---
st.subheader("📊 Mean NDVI per Woreda (2023)")
summary_df = zs_gdf[["admin3Name", "mean"]].rename(columns={
    "admin3Name": "Woreda",
    "mean": "Mean NDVI"
})
st.dataframe(summary_df)

# --- Download button ---
csv = summary_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download NDVI by Woreda (CSV)",
    data=csv,
    file_name="ndvi_by_woreda_2023.csv",
    mime="text/csv"
)