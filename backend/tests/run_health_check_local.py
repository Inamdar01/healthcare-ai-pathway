import importlib.util
from pathlib import Path
import sys

MOD_PATH = Path(__file__).resolve().parents[1] / 'app.py'

spec = importlib.util.spec_from_file_location('backend_app', str(MOD_PATH))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

app = getattr(mod, 'app', None)
if app is None:
    print('Failed to import app')
    sys.exit(2)

client = app.test_client()
resp = client.get('/health')
print('Status code:', resp.status_code)
print('JSON:', resp.get_json())
if resp.status_code == 200:
    print('Health check PASSED')
    sys.exit(0)
else:
    print('Health check FAILED')
    sys.exit(3)
