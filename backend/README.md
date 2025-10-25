# Backend — Health Model (local)

This folder contains a small example model and a tiny Flask app that loads the pickled model and exposes a prediction endpoint.

Quick setup (Windows / PowerShell)

1. Create and activate a virtual environment (recommended):

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

2. Install requirements:

   py -3 -m pip install --upgrade pip
   py -3 -m pip install -r requirements.txt

You can also install globally (not recommended) or use the provided script from the project root to launch both the backend and a static server.

3. Train the model (if you want to re-create it):

   py -3 train_model.py

   This will read `data/diabetes.csv` and save `model.pkl` in this folder.

4. Run the small Flask app (for testing only):

    Option 1 — recommended (keep server running in a dedicated terminal):

       py -3 app.py

    Option 2 — use the project helper to open backend and static server in separate windows (from project root):

       .\scripts\start-dev.ps1

   The app listens on http://127.0.0.1:5000 and provides:
   - GET /model-check — returns JSON indicating whether the model was loaded
   - POST /predict — accepts JSON { "features": [ ... ] } and returns predictions and probabilities

Serving the frontend page

Option A — quick static server (recommended):
  1. In a separate terminal, serve the `template` folder so your browser can load the HTML page from http://127.0.0.1:8000
     - cd ../template
     - py -3 -m http.server 8000
  2. Open http://127.0.0.1:8000/health-risk-predictor.html and use the form. The page POSTs to http://127.0.0.1:5000/predict.

Troubleshooting
- If the frontend shows "Failed to load metadata" ensure:
   - The backend terminal running `py -3 app.py` is open and shows a running server on http://127.0.0.1:5000.
   - You're serving the static page via a local server (Option A) or enabling CORS is allowed in your browser.
   - If you still see connection errors, run the local test to ensure the backend logic is healthy:

      py -3 backend\tests\run_predict_test_local.py

Option B — open the HTML file directly
  - Opening the HTML file via file:// may cause cross-origin restrictions depending on the browser. If fetch requests fail, use Option A or enable CORS on the backend (already included via flask-cors in requirements and app).

Testing the API via curl / PowerShell

Example PowerShell request (replace values with your features):

  $body = '{"features":[0,120,70,20,79,32.0,0.472,33]}'
  Invoke-RestMethod -Method Post -ContentType 'application/json' -Body $body -Uri http://127.0.0.1:5000/predict

Notes & next steps
- For production do not use the built-in Flask server; use a WSGI server (gunicorn/uvicorn/etc.).
- Consider providing feature validation and a small metadata endpoint that documents the feature order so the frontend can render dynamic forms.
