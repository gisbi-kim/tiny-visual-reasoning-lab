"""Publish a few exact experiment inputs with saved predictions; no model inference."""
import base64
import hashlib
import json
import shutil
from io import BytesIO
from pathlib import Path

from PIL import Image
from render import render

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def photo_sections(manifest):
    intro = '''![실제 시연 사진판: 윗줄 개·바나나·고양이, 아랫줄 구명보트·금붕어·배낭](figures/example-canonical.png)

위 사진은 앱과 구어체 시연에서 실제로 사용한 입력이다. 윗줄은 왼쪽부터 A·B·C, 아랫줄은 D·E·F이다. 사진 안에는 사물 이름이나 정답을 넣지 않았다. 아래 표의 이름은 독자를 위한 설명이다.'''
    success = ['''### 실제 사진에서 달라진 선택

숫자로 본 개선을 실제 입력과 연결해 보자. 아래는 각 평가 파일에서 **한국어·정답이 A–F·v1 오답→v2 정답** 조건을 만족하는 첫 사례이다. 이해를 위한 사례이며, 이 세 장만으로 정확도를 계산한 것은 아니다. 질문과 예측은 당시 저장된 기록 그대로이고, 사진도 모델에 입력한 격자 파일을 그대로 복사했다.''']
    for row in manifest['examples']:
        if row['kind'] != 'improved':
            continue
        success.append(f"""#### {row['title']}

![{row['alt']}](figures/{row['file']})

> {row['instruction']}

정답 **{row['gold']}** · v1 **{row['v1']}** → v2 **{row['v2']}**

{row['explanation']} 이 순서는 정답 규칙에 따른 독자용 풀이이며, 모델이 출력한 사고 과정은 아니다.

사진판 `{row['scene_id']}` · [v1 실제 예측](../{row['v1_file']}) · [v2 실제 예측](../{row['v2_file']})""")
    failure = next(row for row in manifest['examples'] if row['kind'] == 'failure')
    failure_text = f"""### 틀린 답도 사진과 함께 보기

![{failure['alt']}](figures/{failure['file']})

> {failure['instruction']}

정답 **{failure['gold']}** · v2 **{failure['v2']}**

{failure['explanation']} 사진판 `{failure['scene_id']}`는 위 목록의 첫 오답과 같은 사례이다. 선택 결과만으로는 사진 인식과 문장 해석 중 어느 단계에서 실패했는지 단정할 수 없다. [실제 예측 기록](../{failure['v2_file']}).

사진 출처는 Stanford Tiny ImageNet이다. 위에 실은 다섯 사진판의 원본 사진 ID, 복사 전후 SHA-256과 사례 선택 기준은 [사진 출처·예측 기록](v2/photo-examples.json)에 남겼다. 사진의 권리는 원 소유자에게 있으며, 프로젝트 코드의 Apache-2.0 라이선스가 사진에 적용되지는 않는다."""
    return intro, '\n\n'.join(success), failure_text


def main():
    figures = ROOT / 'docs/figures'
    figures.mkdir(exist_ok=True)
    keys = ['dog', 'banana', 'cat', 'lifeboat', 'goldfish', 'backpack']
    entries = read('data/demo.json')
    indices = read('docs/demo-selection.json')['indices']
    selected = [[row for row in entries if row['key'] == key][indices[key]] for key in keys]
    assert [r['source_id'] for r in selected] == read('results/v2/freeform.json')['source_ids']
    pictures = []
    for row in selected:
        with Image.open(ROOT / 'data' / row['image']) as im:
            pictures.append(im.convert('RGB'))
    buffer = BytesIO()
    render(pictures).save(buffer, format='PNG')
    data = buffer.getvalue()
    url = 'data:image/png;base64,' + base64.b64encode(data).decode()
    assert hashlib.sha256(url.encode()).hexdigest() == read('docs/v2/api-smoke.json')['scene_data_url_sha256']
    (figures / 'example-canonical.png').write_bytes(data)
    examples = [dict(kind='canonical', file='example-canonical.png', source_ids=[r['source_id'] for r in selected],
                     sha256=hashlib.sha256(data).hexdigest(), matches_recorded_api_scene=True)]
    details = {
        'test_language': ('표현을 바꿔 물은 경우', '윗줄 무당벌레·주전자·키보드, 아랫줄 냉장고·선글라스·피자',
                         '선글라스는 E에 있다. 바로 위 칸 B의 주전자는 주방용품이며, 그 바로 아래에 선글라스가 있으므로 B가 정답이다.'),
        'test_composition': ('조건 두 개를 이어서 물은 경우', '윗줄 거위·트랙터·피망, 아랫줄 자동차·곰·무당벌레',
                            '트랙터 B와 같은 윗줄에서 음식 사진을 찾으면 C의 피망이다. C에서 왼쪽으로 한 칸 이동하면 B가 된다.'),
        'test_combined': ('표현과 조건 조합을 함께 바꾼 경우', '윗줄 버스·프레첼·구명보트, 아랫줄 개·나비·모래시계',
                         '나비는 E에 있다. 같은 아랫줄의 곤충도 E의 나비이므로 E에서 출발한다. 왼쪽으로 한 칸 이동한 D의 개가 정답이다.'),
    }
    for suite, (title, alt, explanation) in details.items():
        old_file = f'results/{suite}_predictions.json'
        new_file = f'results/v2/finetuned/{suite}_predictions.json'
        old, new = read(old_file), read(new_file)
        assert len(old) == len(new)
        for index, (a, b) in enumerate(zip(old, new)):
            assert all(a[k] == b[k] for k in ('scene_id', 'instruction', 'action', 'source_ids'))
            if b['language'] == 'ko' and b['action'] != 'NONE' and a['prediction']['action'] != b['action'] and b['prediction']['action'] == b['action']:
                break
        else:
            raise RuntimeError(f'No matching example: {suite}')
        filename = f'example-{suite}.png'
        original = ROOT / 'data' / b['image']
        shutil.copyfile(original, figures / filename)
        examples.append(dict(kind='improved', title=title, alt=alt, file=filename, suite=suite, row_index=index,
                             scene_id=b['scene_id'], source_ids=b['source_ids'], object_keys=b['object_keys'],
                             input_image=b['image'], sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                             instruction=b['instruction'], gold=b['action'], v1=a['prediction']['action'],
                             v2=b['prediction']['action'], v1_file=old_file, v2_file=new_file, explanation=explanation))
    source = 'results/v2/finetuned/test_language_predictions.json'
    index, row = next((i, r) for i, r in enumerate(read(source)) if r['action'] != r['prediction']['action'])
    original = ROOT / 'data' / row['image']
    filename = 'example-language-failure.png'
    shutil.copyfile(original, figures / filename)
    examples.append(dict(kind='failure', file=filename, row_index=index, scene_id=row['scene_id'],
                         source_ids=row['source_ids'], object_keys=row['object_keys'], input_image=row['image'],
                         sha256=hashlib.sha256(original.read_bytes()).hexdigest(), instruction=row['instruction'],
                         gold=row['action'], v2=row['prediction']['action'], v2_file=source,
                         alt='윗줄 레몬·고양이·금붕어, 아랫줄 콜리플라워·배낭·모래시계',
                         explanation='고양이 B와 같은 윗줄에는 먹거리인 레몬 A가 있다. 따라서 정답은 A인데 v2는 조건에 맞는 사진이 없다는 NONE을 선택했다.'))
    manifest = dict(source_url='https://cs231n.stanford.edu/tiny-imagenet-200.zip',
                    rights='Photographs retain their original owners\' rights; the code license does not apply.',
                    selection='Canonical board matches saved API scene; first Korean non-NONE v1-wrong/v2-correct example per original suite; first v2 error in test_language.',
                    prediction_scope='Previously saved evaluations, no new model execution or dataset changes.', examples=examples)
    (ROOT / 'docs/v2/photo-examples.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('EXACT_INPUT_PHOTO_EXAMPLES_READY', len(examples))


if __name__ == '__main__':
    main()
