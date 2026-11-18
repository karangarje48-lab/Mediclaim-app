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

## Deploy to Streamlit Community Cloud
1. Make sure `data/mediclaim_raw.csv` and
   `model/insurance_model_pipeline.pkl` are included in the GitHub repository.
   The app needs both files at these exact relative paths.
2. Add and push the app and its required files to GitHub:
   ```bash
   git add app.py data/mediclaim_raw.csv model/insurance_model_pipeline.pkl requirements.txt README.md
   git commit -m "Add mediclaim Streamlit app"
   git push
   ```
   Do not upload the local virtual environment.
3. In Streamlit Community Cloud, create an app from that repository, select the
   repository root as the app directory, and set the main file to `app.py`.

The app resolves these files relative to `app.py`, so the working directory
selected by the hosting service does not affect their locations. If a required
file is missing, the app displays the expected path and asks you to include it
before redeploying.
