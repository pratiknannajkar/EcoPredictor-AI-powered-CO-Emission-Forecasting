# CO2 Emission Predictor

A web application for predicting CO2 emissions per capita using machine learning. This project uses a Random Forest Regressor model trained on climate change data and provides an interactive Gradio interface for predictions.

## Features

- 🌍 Predict CO2 emissions per capita based on economic and demographic factors
- 📊 Interactive web interface built with Gradio
- 🤖 Machine learning model using Random Forest Regressor
- 🎯 Automatic feature selection for optimal predictions

## Installation

1. **Clone or download this repository**

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Train the Model

First, train and save the model by running:

```bash
python train_model.py
```

This will:
- Load the cleaned dataset from `Dataset/data_cleaned.csv`
- Train a Random Forest Regressor model
- Perform feature selection
- Save the model to `models/co2_model.pkl`
- Save the feature selector to `models/feature_selector.pkl`

### Step 2: Launch the Gradio App

Once the model is trained, launch the web interface:

```bash
python app.py
```

The app will start and you can access it in your browser (typically at `http://127.0.0.1:7860`).

### Using the Interface

1. Adjust the sliders for various input features:
   - **Economic Indicators**: Cereal yield, FDI, GNI per capita, GDP
   - **Energy & Environment**: Energy use per capita, Protected areas
   - **Population Indicators**: Urban population percentage, Population growth rates

2. Click "Predict CO2 Emissions" to get the prediction

3. View the predicted CO2 emissions per capita in metric tons

## Model Details

- **Algorithm**: Random Forest Regressor
- **Selected Features**: 
  - Energy use per capita
  - Population in urban agglomerations >1M (%)
  - Population growth (annual %)
- **Performance**: R² Score ~0.97, RMSE ~0.77

## Project Structure

```
.
├── app.py                 # Gradio web application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── Dataset/
│   ├── data_cleaned.csv  # Cleaned dataset
│   └── climate_change_download_0.xls  # Original data
└── models/               # Saved models (created after training)
    ├── co2_model.pkl
    ├── feature_selector.pkl
    └── model_info.json
```

## Deployment Options

### Local Deployment
Simply run `python app.py` and access the interface locally.

### Gradio Share Link
Modify `app.py` to set `share=True` in the `launch()` call:
```python
app.launch(share=True)
```
This will create a public shareable link.

### Cloud Deployment
You can deploy this to platforms like:
- **Hugging Face Spaces**: Upload the project to Hugging Face
- **Streamlit Cloud**: Convert to Streamlit or use Gradio's cloud hosting
- **Heroku/Railway**: Deploy as a Python web app

## Requirements

- Python 3.8+
- See `requirements.txt` for package versions

## Notes

- The model was trained on data from 1991-2008
- Predictions are most accurate for values within the training data range
- The model automatically handles feature selection, using only the most important features

## License

This project is for educational purposes as part of a Capstone Project.

