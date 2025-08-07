
import streamlit as st
import geopandas as gpd
import pandas as pd
import leafmap.foliumap as leafmap

st.set_page_config(layout="wide")
st.title("🌿 NDVI Dashboard – Somali Region (Ethiopia)")

# --- File paths ---
excel_file = "ndvi_by_woreda_2023.csv"  # Your CSV file with NDVI values
shapefile = "somali_woredas.geojson"     # Your Somali region boundaries

# --- Load shapefile and NDVI data ---
gdf = gpd.read_file(shapefile)
ndvi_df = pd.read_csv(excel_file)

# --- Merge NDVI values into GeoDataFrame ---
# Assume Excel columns: 'Woreda' and 'Mean NDVI'
gdf = gdf.merge(ndvi_df, left_on="admin3Name", right_on="Woreda")

# --- Display graduated map ---
st.subheader("🗺️ NDVI Choropleth Map by Woreda")

m = leafmap.Map(center=[6.5, 45.5], zoom=6, basemap="CartoDB.Positron")
m.add_data(
    gdf,
    column="Mean NDVI",
    cmap="RdYlGn",
    layer_name="NDVI by Woreda",
    legend_title="Mean NDVI",
    style={"weight": 1, "color": "black", "fillOpacity": 0.7}
)
m.to_streamlit(height=600, width=1200)

# --- NDVI line graph in sidebar ---
st.sidebar.subheader("NDVI by Woreda (Line Graph)")
chart_df = gdf[["Woreda", "Mean NDVI"]].sort_values("Mean NDVI")
st.sidebar.line_chart(chart_df.set_index("Woreda"))

# --- Show summary table ---
st.subheader("📊 Mean NDVI per Woreda (2023)")
st.dataframe(chart_df)

# --- Download button ---
csv = chart_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download NDVI by Woreda (CSV)",
    data=csv,
    file_name="ndvi_by_woreda_2023.csv",
    mime="text/csv"
)
