"""Record public adapter download and real HTTP checks, without loading a model."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def request(path, body=None, port=7861):
    payload = None if body is None else json.dumps(body).encode('utf-8')
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=payload,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['download', 'api'])
    stage = parser.parse_args().stage
    if stage == 'download':
        assert not (ROOT / 'weights-v2').exists(), 'Use a fresh directory for the public download check'
        started = time.perf_counter()
        subprocess.run([sys.executable, 'download_model.py', '--version', 'v2', '--output', 'weights-v2'],
                       cwd=ROOT, check=True)
        manifest = json.loads((ROOT / 'docs/release-v2.json').read_text())
        save('docs/v2/public-download.json', {
            'checked_at': datetime.now(timezone.utc).isoformat(),
            'seconds': time.perf_counter() - started,
            'url': manifest['url'], 'bytes': manifest['bytes'], 'sha256': manifest['sha256'],
            'verified_file_count': len(manifest['files']), 'anonymous_urllib_download': True,
            'fresh_adapter_directory': True, 'base_model_cache_reused': True,
            'python_packages_reused': True, 'physical_mac_tested': False,
        })
        return

    code, data = request('/health')
    health = json.loads(data)
    assert code == 200 and health['version'] == 'v2' and health['ready'], health
    code, data = request('/scene', {'canonical': True})
    assert code == 200
    image = json.loads(data)['image']
    cases = [
        ('개 사진을 골라줘.', 'A'),
        ('바나나 사진을 골라줘.', 'B'),
        ('배낭 사진을 골라줘.', 'F'),
        ('고양이에서 왼쪽으로 한 칸, 그다음 아래로 한 칸 이동한 사진을 골라줘.', 'E'),
        ('고양이의 바로 오른쪽 사진을 골라줘.', 'NONE'),
        ('동물 사진이 3장 이상이면 바나나를, 아니면 배낭을 골라줘.', 'B'),
        ('구명보트와 같은 줄에 있는 동물 사진을 골라줘.', 'E'),
    ]
    rows = []
    for instruction, gold in cases:
        code, data = request('/predict', {'image': image, 'instruction': instruction})
        assert code == 200, data
        prediction = json.loads(data)
        rows.append({'instruction': instruction, 'gold': gold, 'prediction': prediction})
        print(gold, prediction['action'], instruction, flush=True)
    invalid_code, _ = request('/predict', {'image': image, 'instruction': ''})
    random_code, data = request('/scene', {'canonical': False})
    changed = json.loads(data)['image'] != image
    homepage_code, homepage = request('/')
    old_code, _ = request('/', port=7860)
    checks = dict(empty_instruction_status=invalid_code, random_scene_status=random_code,
                  random_scene_changed=changed, homepage_status=homepage_code,
                  homepage_has_v2=b'v2' in homepage, original_mnist_status=old_code)
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), health=health,
                  scene_data_url_sha256=hashlib.sha256(image.encode()).hexdigest(),
                  count=len(rows), correct=sum(r['gold']==r['prediction']['action'] for r in rows),
                  rows=rows, checks=checks)
    save('docs/v2/api-smoke.json', result)
    assert invalid_code == 400 and random_code == homepage_code == old_code == 200
    assert changed and checks['homepage_has_v2']
    print('HTTP_CHECK_COMPLETE', result['correct'], '/', len(rows))


if __name__ == '__main__':
    main()
