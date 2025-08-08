'''
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask
import leafmap.foliumap as leafmap
from rasterstats import zonal_stats
import os
import time
import uuid
import plotly.express as px
import folium
from branca.colormap import linear

# ----------------- STREAMLIT CONFIG -----------------
st.set_page_config(layout="wide")
st.title("Soil Moisture Index (SMI) Dashboard – Somali Region (Ethiopia)")

# ----------------- FILE PATHS -----------------
SHAPEFILE = "somali_woredas.geojson"
SMI_FILES = {
    "2021": "SMAP_L3_2021.tif",
    "2022": "SMAP_L3_2022.tif",
    "2023": "SMAP_L3_2023.tif",
    "2024": "SMAP_L3_2024.tif",
    "2025": "SMAP_L3_2025.tif",
}
MULTI_YEAR_FILE = "SMAP_L3_MultiYearMean_2021_2025.tif"

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("Year Selection")
selected_year = st.sidebar.selectbox(
    "Choose a year to view SMI data",
    options=list(SMI_FILES.keys()),
    index=0
)

show_multi_year = st.sidebar.checkbox(
    "Show Multi-Year Mean (2021–2025) for comparison",
    value=False
)

show_difference = st.sidebar.checkbox(
    "Show Difference Map (Selected Year - Multi-Year Mean)",
    value=False
)

# ----------------- LOAD SHAPEFILE -----------------
# Cache the geodataframe to avoid reloading on every rerun
@st.cache_data
def load_geodataframe(path):
    return gpd.read_file(path).to_crs("EPSG:4326")

gdf = load_geodataframe(SHAPEFILE)

# ----------------- FUNCTION TO MASK RASTER -----------------
# Cache the raster masking to avoid re-processing
@st.cache_data
def mask_raster(raster_path, _shapefile_gdf):
    # Use a unique temporary filename
    output_name = f"masked_{os.path.basename(raster_path)}_{uuid.uuid4()}.tif"
    
    with rasterio.open(raster_path) as src:
        geoms = [geom.__geo_interface__ for geom in _shapefile_gdf.geometry]
        out_image, out_transform = mask(src, geoms, crop=True, nodata=np.nan)
        meta = src.meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": np.nan
        })
        with rasterio.open(output_name, "w", **meta) as dst:
            dst.write(out_image)
            
    # Return the path to the newly created temporary file
    return output_name

# ----------------- LOAD SELECTED YEAR RASTER -----------------
raster_file = SMI_FILES[selected_year]
if not os.path.exists(raster_file):
    st.error(f"Raster file not found: {raster_file}")
    st.stop()

# Pass the gdf directly to the function
masked_tif_year = mask_raster(raster_file, gdf)

# ----------------- LOAD MULTI-YEAR MEAN RASTER (IF NEEDED) -----------------
masked_tif_multi = None
if show_multi_year or show_difference:
    if not os.path.exists(MULTI_YEAR_FILE):
        st.error(f"Multi-year mean file not found: {MULTI_YEAR_FILE}")
        st.stop()
    masked_tif_multi = mask_raster(MULTI_YEAR_FILE, gdf)

# ----------------- CREATE DIFFERENCE RASTER -----------------
@st.cache_data
def create_diff_raster(year_path, multi_path):
    diff_tif = f"smi_difference_{uuid.uuid4()}.tif"
    with rasterio.open(year_path) as year_src, rasterio.open(multi_path) as multi_src:
        year_data = year_src.read(1)
        multi_data = multi_src.read(1)
        diff_data = year_data - multi_data  # Positive = wetter, Negative = drier

        diff_meta = year_src.meta.copy()
        diff_meta.update(nodata=np.nan)

        with rasterio.open(diff_tif, "w", **diff_meta) as dst:
            dst.write(diff_data, 1)
    return diff_tif

diff_tif = None
if show_difference and masked_tif_multi:
    diff_tif = create_diff_raster(masked_tif_year, masked_tif_multi)

# ----------------- ZONAL STATISTICS -----------------
#st.sidebar.header("📊 Zonal Statistics")

# Cache the zonal statistics to improve performance
@st.cache_data
def calculate_zonal_stats(_gdf, raster_path):
    return zonal_stats(
        vectors=_gdf,
        raster=raster_path,
        stats=["mean"],
        geojson_out=True,
        nodata=np.nan
    )

zs_year = calculate_zonal_stats(gdf, masked_tif_year)
zs_gdf_year = gpd.GeoDataFrame.from_features(zs_year)
zs_gdf_year["mean"] = zs_gdf_year["mean"].round(3)

zs_gdf_multi = None
if show_multi_year and masked_tif_multi:
    zs_multi = calculate_zonal_stats(gdf, masked_tif_multi)
    zs_gdf_multi = gpd.GeoDataFrame.from_features(zs_multi)
    zs_gdf_multi["mean"] = zs_gdf_multi["mean"].round(3)
    
# ----------------- INTERACTIVE MAP -----------------
st.subheader(f"Graduated Soil Moisture Index – {selected_year}")
m = leafmap.Map(center=[6.5, 45.5], zoom=6, basemap="CartoDB.Positron")

# Check if the data for the colormap is valid before proceeding
if not zs_gdf_year.empty and 'mean' in zs_gdf_year.columns and zs_gdf_year['mean'].notna().any():
    # Create a color map for the graduated symbology
    smi_colormap = linear.YlOrBr_05.scale(zs_gdf_year['mean'].min(), zs_gdf_year['mean'].max())

    # Define a style function to color the polygons based on the 'mean' value
    def style_woreda_smi(feature):
        return {
            'fillColor': smi_colormap(feature['properties']['mean']),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        }

    m.add_gdf(
        zs_gdf_year,
        layer_name=f"SMI {selected_year} (Graduated)",
        style_function=style_woreda_smi,
        info_mode="on_hover",
    )
    
    # Add the colormap as a legend to the map using add_child
    smi_colormap.caption = "Mean SMI"
    m.add_child(smi_colormap)
else:
    st.warning("No valid SMI data available for graduated map. Displaying a simple base map.")

m.to_streamlit(height=600, width=1200)

# ----------------- CHARTS -----------------
st.subheader("📈 SMI Line Chart")

# Prepare data for the chart
chart_df = zs_gdf_year[["admin3Name", "mean"]].rename(columns={"admin3Name": "Woreda", "mean": f"{selected_year} Mean SMI"})
if show_multi_year and zs_gdf_multi is not None:
    zs_gdf_multi_renamed = zs_gdf_multi[["admin3Name", "mean"]].rename(columns={"admin3Name": "Woreda", "mean": "Multi-Year Mean SMI"})
    chart_df = chart_df.merge(zs_gdf_multi_renamed, on="Woreda")
    
if show_difference and zs_gdf_multi is not None:
    diff_df = zs_gdf_year[["admin3Name", "mean"]].copy()
    diff_df["mean"] = zs_gdf_year["mean"] - zs_gdf_multi["mean"]
    diff_df = diff_df.rename(columns={"admin3Name": "Woreda", "mean": "Difference"})
    chart_df = chart_df.merge(diff_df, on="Woreda")

# Melt the DataFrame for easy plotting with plotly.express
melted_df = chart_df.melt(id_vars="Woreda", var_name="SMI Type", value_name="SMI Value")

fig = px.line(
    melted_df,
    x="Woreda",
    y="SMI Value",
    color="SMI Type",
    title=f"SMI per Woreda ({selected_year} vs. Multi-Year Mean)"
)
st.plotly_chart(fig, use_container_width=True)

# ----------------- SUMMARY TABLES -----------------
if show_multi_year and zs_gdf_multi is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Mean SMI – {selected_year}")
        summary_df_year = zs_gdf_year[["admin3Name", "mean"]].rename(columns={
            "admin3Name": "Woreda",
            "mean": "Mean SMI"
        })
        st.dataframe(summary_df_year)
    with col2:
        st.subheader("Mean SMI (2021–2025)")
        summary_df_multi = zs_gdf_multi[["admin3Name", "mean"]].rename(columns={
            "admin3Name": "Woreda",
            "mean": "Mean SMI"
        })
        st.dataframe(summary_df_multi)
else:
    st.subheader(f"Mean Soil Moisture Index per Woreda – {selected_year}")
    summary_df_year = zs_gdf_year[["admin3Name", "mean"]].rename(columns={
            "admin3Name": "Woreda",
            "mean": "Mean SMI"
        })
    st.dataframe(summary_df_year)

''' 
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
import leafmap.foliumap as leafmap
from rasterstats import zonal_stats
import os
import time
import uuid
import plotly.express as px
import folium
from branca.colormap import linear, StepColormap

# ----------------- STREAMLIT CONFIG -----------------
st.set_page_config(layout="wide")
st.title("💧🌱 SMI and NDVI Dashboard – Somali Region (Ethiopia)")

# ----------------- FILE PATHS -----------------
SHAPEFILE = "somali_woredas.geojson"
# SMI (Soil Moisture Index) Data
SMI_FILES = {
    "2021": "SMAP_L3_2021.tif",
    "2022": "SMAP_L3_2022.tif",
    "2023": "SMAP_L3_2023.tif",
    "2024": "SMAP_L3_2024.tif",
    "2025": "SMAP_L3_2025.tif",
}
# NDVI (Normalized Difference Vegetation Index) Data
NDVI_FILES = {
    "2021": "MODIS_NDVI_Annual_2021.tif",
    "2022": "MODIS_NDVI_Annual_2022.tif",
    "2023": "MODIS_NDVI_Annual_2023.tif",
    "2024": "MODIS_NDVI_Annual_2024.tif",
    "2025": "MODIS_NDVI_Annual_2025.tif",
}

# Multi-year mean files
SMI_MULTI_YEAR_FILE = "SMAP_L3_MultiYearMean_2021_2025.tif"
NDVI_MULTI_YEAR_FILE = "MODIS_NDVI_MultiYearMean_2021_2025.tif"

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("Data Selection")
data_type = st.sidebar.radio("Choose data type:", ["SMI", "NDVI", "Weighted Overlay"])

files_to_use = {}
multi_year_file_to_use = ""
NDVI_SCALE_FACTOR = 10000.0  # Common scale factor for MODIS NDVI

if data_type == "SMI":
    files_to_use = SMI_FILES
    multi_year_file_to_use = SMI_MULTI_YEAR_FILE
elif data_type == "NDVI":
    files_to_use = NDVI_FILES
    multi_year_file_to_use = NDVI_MULTI_YEAR_FILE
elif data_type == "Weighted Overlay":
    files_to_use = SMI_FILES

st.sidebar.header("🔍 Year Selection")
selected_year = st.sidebar.selectbox(
    f"Choose a year to view {data_type} data",
    options=list(files_to_use.keys()),
    index=0
)

if data_type == "Weighted Overlay":
    st.sidebar.header("⚖️ Set Weights for Overlay")
    smi_weight = st.sidebar.slider("SMI Weight", 0.0, 1.0, 0.5, 0.05)
    ndvi_weight = 1.0 - smi_weight
    st.sidebar.markdown(f"NDVI Weight: **{ndvi_weight:.2f}**")

show_multi_year = st.sidebar.checkbox(
    f"Show Multi-Year Mean for comparison",
    value=False
)

show_difference = st.sidebar.checkbox(
    f"Show Difference Map (Selected Year - Multi-Year Mean)",
    value=False
)

# ----------------- LOAD SHAPEFILE -----------------
@st.cache_data
def load_geodataframe(path):
    return gpd.read_file(path).to_crs("EPSG:4326")

gdf = load_geodataframe(SHAPEFILE)

# ----------------- FUNCTION TO MASK RASTER -----------------
@st.cache_data
def mask_raster(raster_path, _shapefile_gdf):
    if not os.path.exists(raster_path):
        return None

    output_name = f"masked_{os.path.basename(raster_path)}_{uuid.uuid4()}.tif"

    with rasterio.open(raster_path) as src:
        geoms = [geom.__geo_interface__ for geom in _shapefile_gdf.geometry]
        out_image, out_transform = mask(src, geoms, crop=True, nodata=src.nodata)
        meta = src.meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": src.nodata if src.nodata is not None else -9999
        })
        with rasterio.open(output_name, "w", **meta) as dst:
            dst.write(out_image)

    return output_name

# ----------------- LOAD RASTER FILES -----------------
raster_file_to_analyze = None
masked_tif_year = None
masked_tif_multi = None

if data_type == "Weighted Overlay":
    smi_raster_file = SMI_FILES[selected_year]
    ndvi_raster_file = NDVI_FILES[selected_year]
    if not os.path.exists(smi_raster_file) or not os.path.exists(ndvi_raster_file):
        st.error(f"SMI or NDVI raster files not found for {selected_year}. Please ensure they exist.")
        st.stop()

    # Mask both rasters
    masked_smi = mask_raster(smi_raster_file, gdf)
    masked_ndvi = mask_raster(ndvi_raster_file, gdf)

    if masked_smi and masked_ndvi:
        # Create a new weighted overlay raster
        overlay_file = f"weighted_overlay_{selected_year}_{uuid.uuid4()}.tif"

        with rasterio.open(masked_smi) as smi_src, rasterio.open(masked_ndvi) as ndvi_src:
            smi_data = smi_src.read(1).astype(np.float32)
            # Scale NDVI data to the correct range (-1 to 1)
            ndvi_data = ndvi_src.read(1).astype(np.float32) / NDVI_SCALE_FACTOR

            # Resample NDVI data to match SMI data if shapes are different
            if smi_data.shape != ndvi_data.shape:
                out_ndvi = np.zeros_like(smi_data)

                reproject(
                    source=ndvi_src.read(1), # Use the raw data for reprojection
                    destination=out_ndvi,
                    src_transform=ndvi_src.transform,
                    src_crs=ndvi_src.crs,
                    dst_transform=smi_src.transform,
                    dst_crs=smi_src.crs,
                    resampling=Resampling.bilinear
                )
                ndvi_data = out_ndvi / NDVI_SCALE_FACTOR # Scale after reprojection

            # Mask out nodata values before calculating
            nodata_mask = np.logical_or(np.isnan(smi_data), np.isnan(ndvi_data))

            # Normalize data to a common scale (e.g., 0-1) to ensure fair weighting
            smi_norm = (smi_data - np.nanmin(smi_data)) / (np.nanmax(smi_data) - np.nanmin(smi_data))
            ndvi_norm = (ndvi_data - np.nanmin(ndvi_data)) / (np.nanmax(ndvi_data) - np.nanmin(ndvi_data))

            overlay_data = (smi_norm * smi_weight) + (ndvi_norm * ndvi_weight)
            overlay_data[nodata_mask] = np.nan

            overlay_meta = smi_src.meta.copy()
            overlay_meta.update(nodata=np.nan)

            with rasterio.open(overlay_file, "w", **overlay_meta) as dst:
                dst.write(overlay_data.astype(overlay_meta['dtype']), 1)
        raster_file_to_analyze = overlay_file

else:
    raster_file = files_to_use[selected_year]
    if not os.path.exists(raster_file):
        st.error(f"Raster file not found: {raster_file}. Please ensure it exists.")
        st.stop()
    raster_file_to_analyze = raster_file

masked_tif_year = mask_raster(raster_file_to_analyze, gdf)
multi_year_file_to_use = SMI_MULTI_YEAR_FILE if data_type == "SMI" else NDVI_MULTI_YEAR_FILE
masked_tif_multi = mask_raster(multi_year_file_to_use, gdf)

# ----------------- CREATE DIFFERENCE RASTER -----------------
@st.cache_data
def create_diff_raster(year_path, multi_path):
    diff_tif = f"diff_raster_{uuid.uuid4()}.tif"
    if not year_path or not multi_path:
        return None

    # CORRECTION: The variable 'multi_path' should be used here, not 'multi_src'
    with rasterio.open(year_path) as year_src, rasterio.open(multi_path) as multi_src:
        year_data = year_src.read(1)
        multi_data = multi_src.read(1)

        # Resample one to match the other if shapes are different
        if year_data.shape != multi_data.shape:
            out_multi = np.zeros_like(year_data)
            reproject(
                source=multi_src.read(1),
                destination=out_multi,
                src_transform=multi_src.transform,
                src_crs=multi_src.crs,
                dst_transform=year_src.transform,
                dst_crs=year_src.crs,
                resampling=Resampling.bilinear
            )
            multi_data = out_multi

        # Scale NDVI data if it's the data type being analyzed
        if data_type == "NDVI":
            year_data = year_data / NDVI_SCALE_FACTOR
            multi_data = multi_data / NDVI_SCALE_FACTOR

        diff_data = year_data - multi_data

        diff_meta = year_src.meta.copy()
        diff_meta.update(nodata=np.nan)

        with rasterio.open(diff_tif, "w", **diff_meta) as dst:
            dst.write(diff_data, 1)
    return diff_tif

diff_tif = None
if show_difference and masked_tif_multi:
    diff_tif = create_diff_raster(masked_tif_year, masked_tif_multi)

# ----------------- ZONAL STATISTICS -----------------
st.sidebar.header("📊 Zonal Statistics")

@st.cache_data
def calculate_zonal_stats(_gdf, raster_path):
    if not raster_path:
        return []

    # Check for NDVI data type and apply scaling before zonal stats
    if data_type == "NDVI" and "MODIS" in raster_path:
        with rasterio.open(raster_path) as src:
            data = src.read(1).astype(np.float32) / NDVI_SCALE_FACTOR
            meta = src.meta.copy()
            meta.update({"dtype": 'float32', "nodata": np.nan})

            temp_path = f"scaled_{os.path.basename(raster_path)}_{uuid.uuid4()}.tif"
            with rasterio.open(temp_path, "w", **meta) as dst:
                dst.write(data, 1)

            stats = zonal_stats(
                vectors=_gdf,
                raster=temp_path,
                stats=["mean"],
                geojson_out=True,
                nodata=np.nan
            )
            os.remove(temp_path)
            return stats

    return zonal_stats(
        vectors=_gdf,
        raster=raster_path,
        stats=["mean"],
        geojson_out=True,
        nodata=np.nan
    )

zs_year = calculate_zonal_stats(gdf, masked_tif_year)
zs_gdf_year = gpd.GeoDataFrame.from_features(zs_year)
if not zs_gdf_year.empty:
    zs_gdf_year["mean"] = zs_gdf_year["mean"].round(3)

zs_gdf_multi = None
if show_multi_year and masked_tif_multi:
    zs_multi = calculate_zonal_stats(gdf, masked_tif_multi)
    zs_gdf_multi = gpd.GeoDataFrame.from_features(zs_multi)
    if not zs_gdf_multi.empty:
        zs_gdf_multi["mean"] = zs_gdf_multi["mean"].round(3)

# ----------------- INTERACTIVE MAP -----------------
st.subheader(f"🗺️ Graduated {data_type} Map – {selected_year}")
m = leafmap.Map(center=[6.5, 45.5], zoom=6, basemap="CartoDB.Positron")

# Check for a valid data range before trying to create a colormap
if not zs_gdf_year.empty and 'mean' in zs_gdf_year.columns and zs_gdf_year['mean'].notna().any():

    # Define a custom color palette for each data type
    if data_type == "SMI":
        colors = ['#fee0d2', '#fc9272', '#de2d26', '#a50f15']
    elif data_type == "NDVI":
        colors = ['#c7e9c0', '#a1d99b', '#41ab5d', '#006d2c']
    else: # Weighted Overlay
        colors = ['#e0f3db', '#a8ddb5', '#43a2ca', '#08589e']

    # Get the data to classify and drop NaN values
    data_to_classify = zs_gdf_year['mean'].dropna()

    # Create 4 quantile-based classes
    try:
        q_labels = pd.qcut(data_to_classify, q=4, labels=False, retbins=True, duplicates='drop')
        bins = q_labels[1].tolist()
        # Ensure the bins cover the full range of the data
        bins[0] = data_to_classify.min()
        bins[-1] = data_to_classify.max()

        # Create a StepColormap with the calculated bins and custom colors
        colormap = StepColormap(
            colors=colors[:len(bins)-1],
            index=bins,
            vmin=bins[0],
            vmax=bins[-1],
            caption=f"Mean {data_type} by Woreda ({selected_year}) - Quantile Classification"
        )

        def style_woreda(feature):
            mean_value = feature['properties']['mean']
            if pd.isna(mean_value):
                return {'fillColor': '#ffffff', 'fillOpacity': 0}
            return {
                'fillColor': colormap(mean_value),
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7
            }

        m.add_gdf(
            zs_gdf_year,
            layer_name=f"{data_type} {selected_year} (Classified)",
            style_function=style_woreda,
            info_mode="on_hover",
        )

        # Add the StepColormap to the map
        m.add_child(colormap)

    except ValueError as e:
        # Handle cases where there aren't enough unique values to create 4 classes
        st.warning(f"Could not create 4 classes for the legend. There might not be enough unique data points. Error: {e}")
        # Fallback to a continuous colormap
        min_val = data_to_classify.min()
        max_val = data_to_classify.max()
        if data_type == "SMI":
            colormap = linear.YlOrBr_05.scale(min_val, max_val)
            colormap.caption = f"Mean SMI ({min_val:.2f} to {max_val:.2f})"
        elif data_type == "NDVI":
            colormap = linear.Greens_05.scale(min_val, max_val)
            colormap.caption = f"Mean NDVI ({min_val:.2f} to {max_val:.2f})"
        else: # Weighted Overlay
            colormap = linear.GnBu_05.scale(min_val, max_val)
            colormap.caption = f"Mean Weighted Overlay ({min_val:.2f} to {max_val:.2f})"

        def style_woreda_continuous(feature):
            mean_value = feature['properties']['mean']
            if pd.isna(mean_value):
                return {'fillColor': '#ffffff', 'fillOpacity': 0}
            return {
                'fillColor': colormap(mean_value),
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7
            }

        m.add_gdf(
            zs_gdf_year,
            layer_name=f"{data_type} {selected_year} (Graduated)",
            style_function=style_woreda_continuous,
            info_mode="on_hover",
        )
        m.add_child(colormap)

else:
    st.warning("No valid data available for graduated map. Displaying a simple base map.")

m.to_streamlit(height=600, width=1200)

# ----------------- CHARTS -----------------
st.subheader(f"📈 {data_type} Line Chart")

chart_df = zs_gdf_year[["admin3Name", "mean"]].rename(columns={"admin3Name": "Woreda", "mean": f"{selected_year} Mean {data_type}"})
if show_multi_year and zs_gdf_multi is not None:
    zs_gdf_multi_renamed = zs_gdf_multi[["admin3Name", "mean"]].rename(columns={"admin3Name": "Woreda", "mean": f"Multi-Year Mean {data_type}"})
    chart_df = chart_df.merge(zs_gdf_multi_renamed, on="Woreda")

if show_difference and zs_gdf_multi is not None:
    diff_df = zs_gdf_year[["admin3Name", "mean"]].copy()
    diff_df["mean"] = zs_gdf_year["mean"] - zs_gdf_multi["mean"]
    diff_df = diff_df.rename(columns={"admin3Name": "Woreda", "mean": "Difference"})
    chart_df = chart_df.merge(diff_df, on="Woreda")

melted_df = chart_df.melt(id_vars="Woreda", var_name=f"{data_type} Type", value_name=f"{data_type} Value")

fig = px.line(
    melted_df,
    x="Woreda",
    y=f"{data_type} Value",
    color=f"{data_type} Type",
    title=f"{data_type} per Woreda ({selected_year} vs. Multi-Year Mean)"
)
st.plotly_chart(fig, use_container_width=True)

# ----------------- SUMMARY TABLES -----------------
if show_multi_year and zs_gdf_multi is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"📋 Mean {data_type} – {selected_year}")
        summary_df_year = zs_gdf_year[["admin3Name", "mean"]].rename(columns={
            "admin3Name": "Woreda",
            "mean": f"Mean {data_type}"
        })
        st.dataframe(summary_df_year)
    with col2:
        st.subheader(f"📋 Multi-Year Mean {data_type} (2021–2025)")
        summary_df_multi = zs_gdf_multi[["admin3Name", "mean"]].rename(columns={
            "admin3Name": "Woreda",
            "mean": f"Mean {data_type}"
        })
        st.dataframe(summary_df_multi)
else:
    st.subheader(f"📋 Mean {data_type} per Woreda – {selected_year}")
    summary_df_year = zs_gdf_year[["admin3Name", "mean"]].rename(columns={
            "admin3Name": "Woreda",
            "mean": f"Mean {data_type}"
        })
    st.dataframe(summary_df_year)

# ----------------- DOWNLOAD BUTTONS -----------------
if not summary_df_year.empty:
    csv = summary_df_year.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download {data_type} by Woreda ({selected_year})",
        data=csv,
        file_name=f"{data_type}_by_woreda_{selected_year}.csv",
        mime="text/csv"
    )

if show_multi_year and zs_gdf_multi is not None and not zs_gdf_multi.empty:
    csv_multi = zs_gdf_multi.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Multi-Year Mean {data_type} (2021–2025)",
        data=csv_multi,
        file_name=f"{data_type}_by_woreda_multi_year_mean.csv",
        mime="text/csv"
    )
