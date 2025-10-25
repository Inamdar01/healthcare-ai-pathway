import importlib.util
import json
import sys
from pathlib import Path

MOD_PATH = Path(__file__).resolve().parents[1] / 'app.py'

spec = importlib.util.spec_from_file_location('backend_app', str(MOD_PATH))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

app = getattr(mod, 'app', None)
if app is None:
    print('Failed to import app from', MOD_PATH)
    sys.exit(2)

client = app.test_client()

features = [0, 120, 70, 20, 79, 32.0, 0.472, 33]
resp = client.post('/predict', json={'features': features})
print('Status code:', resp.status_code)
try:
    data = resp.get_json()
except Exception:
    data = resp.data.decode('utf-8')
print('Response:', json.dumps(data, indent=2) if isinstance(data, dict) else data)

if resp.status_code == 200 and isinstance(data, dict) and 'predictions' in data:
    print('Local test PASSED')
    sys.exit(0)
else:
    print('Local test FAILED')
    sys.exit(3)
