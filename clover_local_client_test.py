import json, os, time, urllib.request, urllib.error
from pathlib import Path

URL = os.environ.get('CLOVER_ENDPOINT', 'http://127.0.0.1:8322/v1/chat/completions')
OUT = Path(os.environ.get('CLOVER_RESULTS_DIR', 'benchmark-results'))
OUT.mkdir(parents=True, exist_ok=True)

def post(payload, timeout=600):
    raw = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(URL, data=raw, headers={'Content-Type': 'application/json; charset=utf-8'}, method='POST')
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        status = resp.status
    elapsed = time.perf_counter() - started
    return status, elapsed, json.loads(body.decode('utf-8'))

def stream(payload, timeout=600):
    raw = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(URL, data=raw, headers={'Content-Type': 'application/json; charset=utf-8'}, method='POST')
    started = time.perf_counter()
    first = None
    pieces = []
    status = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = resp.status
        for line in resp:
            if not line.startswith(b'data: '):
                continue
            item = line[6:].strip()
            if item == b'[DONE]':
                continue
            obj = json.loads(item.decode('utf-8'))
            if first is None:
                first = time.perf_counter() - started
            for choice in obj.get('choices', []):
                delta = choice.get('delta') or {}
                if delta.get('content'):
                    pieces.append(delta['content'])
    elapsed = time.perf_counter() - started
    text = ''.join(pieces)
    return {'status': status, 'wall_s': elapsed, 'first_event_s': first, 'text': text, 'pieces': len(pieces)}

base = {'messages': [{'role': 'user', 'content': '请简洁回答：太阳为什么会发光？'}], 'max_tokens': 16, 'stream': False}
rows = []
for block in (4, 32):
    payload = dict(base, block_size=block)
    for repeat in range(2):
        status, elapsed, response = post(payload)
        timing = response.get('timings', {})
        rows.append({'kind': 'nonstream', 'block_size': block, 'repeat': repeat + 1,
                     'status': status, 'wall_s': elapsed, 'response': response,
                     'effective_tok_per_s': timing.get('tok_per_s')})
stream_row = stream(dict(base, block_size=4, stream=True))
rows.append({'kind': 'stream', 'block_size': 4, **stream_row})
result = {'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'endpoint': URL, 'tests': rows}
out = OUT / 'http-benchmark-20261028.json'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
print('saved=' + str(out))
