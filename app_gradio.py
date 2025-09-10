import gradio as gr
import joblib
import pandas as pd

# --- Load Model Artifacts ---
# This block loads the model, scaler, and columns when the application starts.
try:
    model = joblib.load('logistic_model.joblib')
    scaler = joblib.load('scaler.joblib')
    model_columns = joblib.load('model_columns.joblib')
    print("--- Model artifacts loaded successfully for Gradio app ---")
except Exception as e:
    print(f"--- Error loading model artifacts: {e} ---")
    model = None

# --- Define the Prediction Function ---
# Gradio works by wrapping a standard Python function in a UI.
# This function takes all the user inputs and returns a prediction.
def predict_deforestation(
    lat, lon, elev, slope, rain, temp, cloud, fire,
    ndvi, ndmi, evi, tree_cover, canopy_h, road_dist,
    settlement_dist, river_dist, protected, logging,
    pop_density, forest_loss_3y, cumulative_loss,
    region, country
):
    if not all([model, scaler, model_columns]):
        return "Model artifacts not loaded. Check server logs."

    # 1. Create a dictionary from the inputs to match the original column names
    input_data = {
        'Latitude': lat, 'Longitude': lon, 'Elevation (m)': elev, 'Slope (°)': slope,
        'Rainfall (mm)': rain, 'Temperature (°C)': temp, 'Cloud Cover (%)': cloud,
        'Fire Alerts (7d)': fire, 'NDVI (Vegetation Index)': ndvi, 'NDMI (Moisture Index)': ndmi,
        'EVI (Enhanced Vegetation Index)': evi, 'Tree Cover (%)': tree_cover,
        'Canopy Height (m)': canopy_h, 'Distance to Road (km)': road_dist,
        'Distance to Settlement (km)': settlement_dist, 'Distance to River (km)': river_dist,
        '"Protected Area (1=Yes,0=No)"': protected, '"Logging Concession (1=Yes,0=No)"': logging,
        'Population Density (per km²)': pop_density, 'Forest Loss Last 3Y (%)': forest_loss_3y,
        'Cumulative Deforested Area (%)': cumulative_loss, 'Region': region, 'Country': country
    }

    # 2. Convert to DataFrame
    query_df = pd.DataFrame([input_data])
    
    # 3. Preprocess the data (same as in the Flask app)
    query_processed = pd.get_dummies(query_df, columns=['Region', 'Country'])
    query_aligned = query_processed.reindex(columns=model_columns, fill_value=0)
    query_scaled = scaler.transform(query_aligned)

    # 4. Make prediction
    prediction_proba = model.predict_proba(query_scaled)
    
    # 5. Format the output for Gradio's Label component
    # The output is a dictionary of labels and their confidence scores.
    confidences = {
        'No Deforestation': prediction_proba[0][0],
        'Deforestation': prediction_proba[0][1]
    }
    return confidences

# --- Define the Gradio Interface ---
# This creates the web UI components. Each input corresponds to an argument in the function above.
with gr.Blocks() as iface:
    gr.Markdown("# 🌳 Deforestation Risk Prediction")
    gr.Markdown("Enter the data for a specific geographical tile to predict the risk of a deforestation event.")
    
    with gr.Row():
        region = gr.Dropdown(['Amazon', 'Congo Basin', 'SE Asia'], label="Region")
        country = gr.Dropdown(['Brazil', 'Colombia', 'Peru', 'Congo', 'DRC', 'Gabon', 'Indonesia', 'Malaysia', 'Papua New Guinea'], label="Country")
        protected = gr.Radio([0, 1], label='"Protected Area (1=Yes,0=No)"')
        logging = gr.Radio([0, 1], label='"Logging Concession (1=Yes,0=No)"')

    with gr.Row():
        lat = gr.Number(label="Latitude")
        lon = gr.Number(label="Longitude")
        elev = gr.Number(label="Elevation (m)")
        slope = gr.Number(label="Slope (°)")

    with gr.Row():
        rain = gr.Number(label="Rainfall (mm)")
        temp = gr.Number(label="Temperature (°C)")
        cloud = gr.Number(label="Cloud Cover (%)")
        fire = gr.Number(label="Fire Alerts (7d)")
    
    with gr.Row():
        ndvi = gr.Number(label="NDVI (Vegetation Index)")
        ndmi = gr.Number(label="NDMI (Moisture Index)")
        evi = gr.Number(label="EVI (Enhanced Vegetation Index)")
        tree_cover = gr.Number(label="Tree Cover (%)")
        
    with gr.Row():
        canopy_h = gr.Number(label="Canopy Height (m)")
        road_dist = gr.Number(label="Distance to Road (km)")
        settlement_dist = gr.Number(label="Distance to Settlement (km)")
        river_dist = gr.Number(label="Distance to River (km)")

    with gr.Row():
        pop_density = gr.Number(label="Population Density (per km²)")
        forest_loss_3y = gr.Number(label="Forest Loss Last 3Y (%)")
        cumulative_loss = gr.Number(label="Cumulative Deforested Area (%)")

    predict_btn = gr.Button("Predict Risk", variant="primary")
    output_label = gr.Label(num_top_classes=2, label="Prediction Result")
    
    predict_btn.click(
        fn=predict_deforestation,
        inputs=[
            lat, lon, elev, slope, rain, temp, cloud, fire,
            ndvi, ndmi, evi, tree_cover, canopy_h, road_dist,
            settlement_dist, river_dist, protected, logging,
            pop_density, forest_loss_3y, cumulative_loss,
            region, country
        ],
        outputs=output_label
    )

# --- Launch the App ---
# share=True creates a public, temporary URL for you to share with others.
iface.launch(share=True)