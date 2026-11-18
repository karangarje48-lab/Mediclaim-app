# Mediclaim Insurance Charges Predictor

Streamlit app that predicts annual medical insurance charges using a Random Forest model.

## Folder structure
```
mediclaim-app/
├── venv/                 # virtual environment (create locally, not uploaded to GitHub)
├── app.py                # Streamlit app
├── Mediclaim.ipynb       # data cleaning, EDA, model training
├── data/
│   └── mediclaim_raw.csv
├── model/
│   └── insurance_model_pipeline.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## Run locally
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Push to GitHub (venv/ is ignored) -> share.streamlit.io -> New app -> main file: `app.py`.
