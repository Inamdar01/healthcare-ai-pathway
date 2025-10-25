import json
import sys
import time
from urllib import request as urllib_request


def post_json(url, data):
    data_bytes = json.dumps(data).encode('utf-8')
    req = urllib_request.Request(url, data=data_bytes, headers={
        'Content-Type': 'application/json'
    })
    with urllib_request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode('utf-8'))


def main():
    url = 'http://127.0.0.1:5000/predict'
    # sample feature vector matching FEATURE_NAMES in the app
    features = [0, 120, 70, 20, 79, 32.0, 0.472, 33]
    print('Posting to', url)
    try:
        out = post_json(url, {'features': features})
        print('Response:', json.dumps(out, indent=2))
        if 'predictions' in out:
            print('Test PASSED')
            sys.exit(0)
        else:
            print('Test FAILED: no predictions key')
            sys.exit(2)
    except Exception as e:
        print('Test FAILED, request error:', e)
        sys.exit(3)


if __name__ == '__main__':
    main()
