import os
import pickle
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import json

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model.pkl')

app = Flask(__name__)
CORS(app)

model = None
model_loaded = False
model_info = {}

# Defaults placeholder (populated from dataset if available)
DEFAULTS = {}

# Feature metadata for the diabetes model (order matters)
FEATURE_NAMES = [
	"Pregnancies",
	"Glucose",
	"BloodPressure",
	"SkinThickness",
	"Insulin",
	"BMI",
	"DiabetesPedigreeFunction",
	"Age",
]


def metadata():
	# include computed ranges if available
	ranges = DEFAULTS.get('__ranges__') if isinstance(DEFAULTS, dict) and '__ranges__' in DEFAULTS else {}
	# return defaults without the internal ranges key
	defaults_only = {k: v for k, v in DEFAULTS.items() if k != '__ranges__'} if isinstance(DEFAULTS, dict) else DEFAULTS
	return {"feature_names": FEATURE_NAMES, "num_features": len(FEATURE_NAMES), "defaults": defaults_only, "ranges": ranges}


def compute_defaults():
	"""Compute per-feature default values (means) from backend/data/diabetes.csv if available."""
	defaults = {}
	data_path = os.path.join(base_dir, 'data', 'diabetes.csv')
	if not os.path.exists(data_path):
		# no dataset available
		for feat in FEATURE_NAMES:
			defaults[feat] = None
		return defaults

	try:
		df = pd.read_csv(data_path)
		# helper to normalize column/feature names for matching
		def norm(s):
			return ''.join(ch.lower() for ch in str(s) if ch.isalnum())

		cols_norm = {norm(c): c for c in df.columns}
		# compute mean, min and max per feature when available
		ranges = {}
		for feat in FEATURE_NAMES:
			key = norm(feat)
			if key in cols_norm:
				col = cols_norm[key]
				try:
					series = pd.to_numeric(df[col], errors='coerce').dropna()
					val = float(series.mean()) if not series.empty else None
					mn = float(series.min()) if not series.empty else None
					mx = float(series.max()) if not series.empty else None
					defaults[feat] = round(val, 6) if val is not None else None
					ranges[feat] = {"min": round(mn, 6) if mn is not None else None, "max": round(mx, 6) if mx is not None else None}
				except Exception:
					defaults[feat] = None
					ranges[feat] = {"min": None, "max": None}
			else:
				defaults[feat] = None
				ranges[feat] = {"min": None, "max": None}
		# attach ranges inside defaults under a reserved key so existing code can still use DEFAULTS
		defaults['__ranges__'] = ranges
	except Exception:
		for feat in FEATURE_NAMES:
			defaults[feat] = None
	return defaults


@app.route('/metadata', methods=['GET'])
def get_metadata():
	"""Dedicated metadata endpoint returning feature names and counts."""
	return jsonify(metadata())


@app.route('/health', methods=['GET'])
def health_check():
	"""Lightweight health endpoint for quick browser checks."""
	return jsonify({"status": "ok", "model_loaded": model_loaded, "model_type": model_info.get("model_type")}), 200


def load_model():
	global model, model_loaded, model_info
	try:
		with open(model_path, 'rb') as f:
			model = pickle.load(f)
		model_loaded = True
		model_info = {"model_path": model_path, "model_type": type(model).__name__}
		print(f"✅ Loaded model from {model_path} (type={model_info['model_type']})")
	except Exception as e:
		print(f"❌ Failed to load model at {model_path}: {e}")
		traceback.print_exc()
		model_loaded = False


load_model()

# Populate DEFAULTS from dataset (non-blocking)
try:
	DEFAULTS = compute_defaults()
except Exception:
	DEFAULTS = {feat: None for feat in FEATURE_NAMES}


@app.route('/model-check', methods=['GET'])
def model_check():
	return jsonify({"loaded": model_loaded, **model_info, **metadata()})


@app.route('/predict', methods=['POST'])
def predict():
	if not model_loaded:
		return jsonify({"error": "Model not loaded"}), 500
	data = request.get_json()
	# Persist raw incoming JSON for debugging (helps diagnose client payloads)
	try:
		with open(os.path.join(base_dir, 'debug_predict.log'), 'a', encoding='utf-8') as _f:
			_f.write(json.dumps(data, default=str) + "\n")
	except Exception:
		pass
	print('--- /predict received body:', data)
	# request.get_json() returns None when there is no JSON body or invalid JSON.
	if data is None:
		return jsonify({"error": "Expected JSON body (JSON required). Provide 'features' or 'inputs' payload."}), 400
	# Backwards compatible: accept either a raw 'features' list (old clients)
	# or a structured payload with named inputs (recommended).
	features_list = data.get('features') if isinstance(data, dict) else None
	# If neither legacy 'features' nor structured 'inputs' (or top-level simple keys) are present,
	# return a helpful error rather than failing silently.
	if features_list is None and not (isinstance(data.get('inputs'), dict) or any(k for k in (data.keys() if isinstance(data, dict) else []) if k not in ('features','inputs'))):
		return jsonify({"error": "Missing required payload. Provide either 'features' (array) or 'inputs' (object) with named values."}), 400
	if features_list is not None:
		# Old-style payload
		if not isinstance(features_list, (list, tuple)):
			return jsonify({"error": "'features' must be a list or tuple"}), 400
		if len(features_list) != len(FEATURE_NAMES):
			return jsonify({
				"error": "Invalid feature vector length",
				"expected": len(FEATURE_NAMES),
				"received": len(features_list),
			}), 400
		try:
			numeric = [float(x) for x in features_list]
		except Exception:
			return jsonify({"error": "All features must be numeric"}), 400
		X = np.array(numeric)
		if X.ndim == 1:
			X = X.reshape(1, -1)
	else:
		# New-style payload: read named inputs and compute BMI & pedigree mapping server-side
		inputs = {}
		# accept either an 'inputs' dict or top-level keys
		if isinstance(data.get('inputs'), dict):
			inputs = data.get('inputs')
		else:
			# copy all top-level simple keys (string/number/bool) into inputs
			for k, v in data.items():
				if k in ('features', 'inputs'):
					continue
				if isinstance(v, (str, int, float, bool)):
					inputs[k] = v

		# Build numeric feature vector in the order of FEATURE_NAMES
		# Server-side validation: ensure reasonable height/weight when provided
		h_cm_val = None
		if 'height_cm' in inputs:
			try:
				h_cm_val = float(inputs.get('height_cm'))
			except Exception:
				h_cm_val = None
		elif 'height_m' in inputs:
			try:
				h_cm_val = float(inputs.get('height_m')) * 100.0
			except Exception:
				h_cm_val = None
		w_kg_val = None
		if 'weight_kg' in inputs:
			try:
				w_kg_val = float(inputs.get('weight_kg'))
			except Exception:
				w_kg_val = None

		# sensible server-side ranges
		if h_cm_val is not None and not (100.0 <= h_cm_val <= 250.0):
			return jsonify({"error": "Invalid height_cm", "details": "Height must be between 100 and 250 cm"}), 400
		if w_kg_val is not None and not (30.0 <= w_kg_val <= 300.0):
			return jsonify({"error": "Invalid weight_kg", "details": "Weight must be between 30 and 300 kg"}), 400

		numeric = []
		for feat in FEATURE_NAMES:
			# BMI: compute from height_cm/height_m and weight_kg if provided
			if feat == 'BMI':
				# prefer height_cm and weight_kg keys
				h_cm = None
				w_kg = None
				if 'height_cm' in inputs:
					h_cm = inputs.get('height_cm')
				elif 'height_m' in inputs:
					h_cm = float(inputs.get('height_m')) * 100.0 if inputs.get('height_m') is not None else None
				# weight
				if 'weight_kg' in inputs:
					w_kg = inputs.get('weight_kg')
				# fallback to feature name if present (unlikely)
				if w_kg is None and feat in inputs:
					w_kg = inputs.get(feat)
				try:
					h = float(h_cm) / 100.0 if h_cm is not None else None
					w = float(w_kg) if w_kg is not None else None
					bmi = 0.0
					if h and h > 0 and w is not None:
						bmi = w / (h * h)
				except Exception:
					bmi = 0.0
				numeric.append(float(round(bmi, 2)))
				continue
			# DiabetesPedigreeFunction: map family_history yes/no to numeric, or accept numeric value if provided
			if feat == 'DiabetesPedigreeFunction':
				val = inputs.get('family_history') if 'family_history' in inputs else inputs.get(feat)
				if val is None:
					# fallback to default if available
					val_num = DEFAULTS.get(feat)
					numeric.append(float(val_num) if val_num is not None else 0.0)
					continue
				# if client provided numeric, use it
				try:
					valf = float(val)
					numeric.append(valf)
					continue
				except Exception:
					# map common string answers
					if isinstance(val, str) and val.lower() in ('yes', 'y', 'true', '1'):
						numeric.append(0.5)
					elif isinstance(val, str) and val.lower() in ('no', 'n', 'false', '0'):
						numeric.append(0.2)
					else:
						# fallback to heuristic
						numeric.append(0.2)
					continue

			# default: try to read the feature directly from inputs or defaults
			v = inputs.get(feat)
			if v is None:
				v = DEFAULTS.get(feat)
			try:
				numeric.append(float(v) if v is not None else 0.0)
			except Exception:
				numeric.append(0.0)

		# final numeric array
		X = np.array(numeric)
		if X.ndim == 1:
			X = X.reshape(1, -1)

	# Run prediction
	try:
		preds = model.predict(X).tolist()
		proba = None
		if hasattr(model, 'predict_proba'):
			proba = model.predict_proba(X).tolist()
		return jsonify({"predictions": preds, "probabilities": proba})
	except Exception as e:
		return jsonify({"error": "prediction failed", "details": str(e)}), 500


if __name__ == '__main__':
	# For local testing only. Use a real WSGI server for production.
	# Bind to 0.0.0.0 so local static servers or other local tools can reach the backend.
	# Disable Flask reloader to make the server process stable when launched from tooling.
	app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
