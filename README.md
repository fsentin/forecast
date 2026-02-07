# Time Series Forecasting App

Interactive app for time series forecasting with data preprocessing, multiple models with hyperparameter settings and model comparison.


## Key Features

- **Three Forecasting Models Implemented**: ARIMA (statistical), Prophet (business-focused), N-BEATS (deep learning)
- **Easy Model Addition**: Responsive UI for easy new model implementation
- **Smart Recommendations**: Models can provide hyperparameter recommendations based on time series qualities
- **Data Preprocessing**: Automatic gap detection/filling and outlier removal
- **Model Comparison**: Side-by-side evaluation with MAE and RMSE metrics
- **Easy Metric Addition**: Responsive UI for easy new metric implementation
- **Interactive UI**: Clean interface with interactive visualizations

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/forecast.git
cd forecast

# Create virtual environment 
python -m venv venv
source venv/bin/activate  # Linux
source venv\Scripts\activate  # Windows

# OR use conda
conda create -n forecast python=3.12
conda activate forecast

# Install and run
pip install -r requirements.txt
streamlit run app.py
```



## Usage flow

1. Upload CSV with date and value columns
2. Select columns and apply preprocessing
3. Choose a model tab (ARIMA/Prophet/N-BEATS)
4. Configure hyperparameters (or use recommendations)
5. Specify forecast horizon and evaluation train-test split
6. Train model
7. Compare trained models in the comparison tab


## Expected Data Upload Format

User can upload CSV containing multiple columns, but must include:
- At least one **datetime column** (any standard format)
- At least one **numeric value column** to forecast

Example:
```csv
date,sales,temperature,value
2024-01-01,100,15,234
2024-01-02,105,16,241
2024-01-03,102,14,238
```
User can select which columns to use for forecasting during upload.