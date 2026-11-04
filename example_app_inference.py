"""로컬 앱과 같은 추론 엔진을 직접 호출한다."""
import json
from pathlib import Path

from PIL import Image

from engine import DecisionEngine


def main():
    image_path = Path('docs/figures/example-canonical.png')
    with Image.open(image_path) as source:
        image = source.convert('RGB')

    instruction = '고양이에서 왼쪽으로 한 칸, 그다음 아래로 한 칸 이동한 사진을 골라줘.'
    engine = DecisionEngine(checkpoint='weights-v2', device='auto')
    result = engine.predict(image, instruction)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
