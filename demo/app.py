import joblib
import pandas as pd
import gradio as gr
from huggingface_hub import hf_hub_download

#download our model from hugging face and load it
path = hf_hub_download(repo_id="iaac-maai/day2-team6-energy-use", filename="model.joblib")
model = joblib.load(path)

#the categories the model knows
encoder = model.named_steps["preprocessing"].named_transformers_["nom"].named_steps["encoder"]
uses, building_types, neighborhoods = [list(values) for values in encoder.categories_]

#median coordinates of each neighborhood in the data
coordinates = {
    "BALLARD": (47.669, -122.380),
    "CENTRAL": (47.608, -122.308),
    "DELRIDGE": (47.536, -122.364),
    "DOWNTOWN": (47.610, -122.336),
    "EAST": (47.616, -122.323),
    "GREATER DUWAMISH": (47.568, -122.327),
    "LAKE UNION": (47.635, -122.337),
    "MAGNOLIA / QUEEN ANNE": (47.630, -122.358),
    "NORTH": (47.713, -122.313),
    "NORTHEAST": (47.663, -122.312),
    "NORTHWEST": (47.706, -122.346),
    "SOUTHEAST": (47.552, -122.282),
    "SOUTHWEST": (47.563, -122.387),
}


def predict(use, building_type, neighborhood, year_built, floors, gfa_m2, parking_share):
    floors = max(int(floors), 1)
    latitude, longitude = coordinates[neighborhood]
    building = pd.DataFrame({
        "primary_property_type": [use],
        "building_type": [building_type],
        "neighborhood": [neighborhood],
        "year_built": [year_built],
        "number_of_floors": [floors],
        "number_of_buildings": [1],
        "gfa_m2": [gfa_m2],
        "gfa_per_floor_m2": [gfa_m2 / floors],
        "parking_share": [parking_share],
        "latitude": [latitude],
        "longitude": [longitude],
    })
    eui = 10 ** model.predict(building)[0]
    return (f"{eui:.0f} kWh/m2 per year\n"
            f"{eui * gfa_m2:,.0f} kWh per year for the whole building\n"
            f"usually somewhere between {eui / 1.45:.0f} and {eui * 1.45:.0f} kWh/m2 (typical error around 45%)")


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(uses, value="Mid-Rise Multifamily", label="Use"),
        gr.Dropdown(building_types, value="Multifamily MR (5-9)", label="City building type"),
        gr.Dropdown(neighborhoods, value="EAST", label="Neighborhood"),
        gr.Number(2016, label="Year built", precision=0),
        gr.Number(6, label="Floors", precision=0),
        gr.Number(6000, label="Floor area (m2)"),
        gr.Slider(0, 0.8, value=0.1, step=0.05, label="Parking share"),
    ],
    outputs=gr.Textbox(label="Estimated energy use", lines=3),
    title="Seattle building energy estimate",
    description="A first, rough estimate of the yearly energy use per m2 of a building in Seattle, from a random forest trained on the 2016 Seattle building energy data. Not an exact number.",
    flagging_mode="never",
)

demo.launch()
