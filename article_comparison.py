"""Add the user's screenshots and measured paired comparison to the article."""
import collections, json, math, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def add(text):
    read=lambda name:json.loads((ROOT/'results/generative-comparison'/name).read_text(encoding='utf-8'))
    d,g=read('decision.json'),read('generative.json')
    assert d['count']==g['count']==96
    for a,b in zip(d['rows'],g['rows']):
        assert all(a[k]==b[k] for k in ('scene_id','instruction','gold','image_sha256','family','language'))
    def relaxed(raw):
        match=re.fullmatch(r'\s*(?:정답|답|answer)\s*[:：]\s*(NONE|[A-F])\s*[.!]?\s*',raw,re.I)
        return match.group(1).upper() if match else None
    flexible=sum((r['action'] or relaxed(r['raw']))==r['gold'] for r in g['rows'])
    unique=len({r['scene_id'] for r in d['rows']})
    fast=g['median_ms']/d['median_ms']
    out=dict(count=96,decision_correct=d['correct'],generative_strict_correct=g['correct'],generative_flexible_correct=flexible,
             generative_invalid_strict=g['invalid'],decision_median_ms=d['median_ms'],generative_median_ms=g['median_ms'],
             generative_over_decision_median_ratio=fast,unique_scenes=unique,identical_input_rows_verified=True,
             sample='seed 3407; four rows per family and language from existing v2_test; 12 families, two languages',
             timing='RTX 4070 Ti SUPER; 4bit NF4/BF16; image open and model load excluded; warmups 3; CUDA sync; processor to output',
             generative='Qwen/Qwen3.5-4B fixed base revision; no LoRA or decision head; greedy; thinking disabled via chat template; max 8 new tokens',
             decision='v2 fine-tuned LoRA and Clef decision head',
             parsing='Strict: only A-F or NONE plus optional period. Relaxed additionally accepts answer: B / 정답: B.',
             caveat='Both adaptation and output mechanism differ; results depend on prompt and token budget, and are not a controlled head-only comparison.')
    (ROOT/'results/generative-comparison/summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    section=f'''## 생성형 Qwen에게 같은 문제를 물어보면?

앞에서 선택 점수를 한 번 계산하는 방식을 소개했다. 그렇다면 같은 Qwen 4B가 답을 글자로 직접 쓰게 하면 어떨까? 실제로 두 경로를 같은 사진과 질문으로 실행해 봤다. 학습에서 제외한 사진으로 만든 기존 `v2_test`에서 12개 작업 유형 × 한국어·영어 각각 네 문제씩, 총 **96문제**를 고정 seed로 뽑았다. 각 문제의 사진 파일 해시·질문·정답이 두 결과에서 일치하는지도 확인했다.

한쪽은 **기본 Qwen3.5-4B**에게 A–F 또는 NONE 하나만 출력하라고 요청하는 생성형 경로다. 다른 한쪽은 같은 기본 모델에 학습한 LoRA와 선택용 head를 붙인 **v2 decision model**이다. 같은 Windows의 RTX 4070 Ti SUPER에서 두 모델을 차례로 실행하고, 세 문제로 준비 운동을 한 뒤 한 문제씩 시간을 쟀다. 둘 다 NF4 4bit/BF16을 사용했다.

| 같은 96문제 | 생성형 기본 Qwen | 학습한 decision model v2 |
|---|---:|---:|
| 정확한 선택 | {g['correct']}/96 ({100*g['correct']/96:.2f}%) | {d['correct']}/96 ({100*d['correct']/96:.2f}%) |
| 단건 시간 중앙값 | {g['median_ms']:.1f}ms | {d['median_ms']:.1f}ms |

이번 설정에서 중앙값은 생성형이 약 **{fast:.2f}배** 오래 걸렸다. 두 시간 모두 모델을 미리 올리고 이미지를 연 뒤, 전처리부터 결과 변환까지 포함해 GPU 작업 완료를 기다린 값이다. 모델 로딩·디스크에서 이미지 열기·HTTP와 네트워크는 빠져 있다. 같은 사진판이 여러 문제에 쓰일 수 있으므로 96개를 서로 독립적인 사진 96장으로 생각하면 안 된다.

생성형에는 생각 모드를 끄고, 무작위 샘플링 없이 최대 8토큰으로 답하도록 했다. A–F 또는 NONE만 나온 경우를 읽었더니 형식이 벗어난 답이 {g['invalid']}개였다. 그중 “정답: B”는 사람이 읽을 수 있는 답이라 이 형식까지 허용하면 **{flexible}/96 ({100*flexible/96:.2f}%)**가 된다. 나머지 세 답은 짧은 토큰 한도 안에서 설명을 시작했다. 원문은 결과 파일에 그대로 남겼다.

생성형에는 파인튜닝을 하지 않았고 v2에는 추가 학습을 했으므로 정확도 차이를 출력 head만의 효과로 볼 수는 없다. 프롬프트와 출력 길이를 바꾸면 생성형의 성능과 시간도 달라질 수 있다. 그래도 이 설정에서는 원하는 답을 바로 전달하는 선택 방식의 속도와, 작업 데이터를 추가했을 때의 정확도를 함께 확인할 수 있었다. [측정 요약](../results/generative-comparison/summary.json) · [생성형의 실제 출력](../results/generative-comparison/generative.json) · [decision model의 실제 출력](../results/generative-comparison/decision.json) · [실험 코드](../compare_generation.py).

'''
    reasoning_files=[ROOT/f'results/generative-comparison/thinking-{int(t)}-tokens-{n}.json' for t,n in [(False,40),(True,40),(False,120),(True,120)]]
    if all(path.exists() for path in reasoning_files):
        from reasoning_article import create
        section=create()
    text=text.replace('## 11. 가중치의 출처와 다시 실행하기',section+'## 11. 가중치의 출처와 다시 실행하기',1)
    text=text.replace('같은 조건에서 생성형 Qwen의 답변 시간은 재지 않았으므로, 이 숫자만으로 몇 배 빠르다고 말할 수는 없다. 다만 선택 점수만 계산하는 구조가 왜 필요한지는 짐작할 수 있다.', '이 수치를 기억해 두자. 뒤에서 같은 96문제를 생성형 Qwen과 decision model에 각각 풀려 보고 시간도 직접 비교한다.')
    text=text.replace('다만 이번 실험은 그 기여를 분리해 측정한 실험이 아니며, 파인튜닝 전 순수 Qwen 대조군을 실행하지 않았다.', '다만 초기 실험만으로는 사전학습의 기여를 분리하기 어렵다. 뒤에서 순수 Qwen도 같은 문제의 일부에 실행해 비교한다.')
    snapshots='''### 직접 입력해 본 세 화면

같은 사진판에서 사용자가 동물 사진의 기준을 1장, 2장, 3장으로 바꿔 본 화면이다. A에는 초록색 음식, B에는 금붕어, E에는 동물 사진이 보인다. 기준이 1장이나 2장일 때는 적어도 B와 E 두 장이 동물이므로 A가 자연스러운 답이다. 화면에는 두 경우 모두 F가 표시되었다. 세 번째 화면에는 NONE이 표시되어 있다.

![사용자 직접 테스트: 1장 이상이라는 지시와 F 선택 화면](figures/user-test-1.png)

![사용자 직접 테스트: 2장 이상이라는 지시와 F 선택 화면](figures/user-test-2.png)

![사용자 직접 테스트: 3장 이상이라는 지시와 NONE 선택 화면](figures/user-test-3.png)

첫 화면에서 F의 표시 확률은 96.1%, 두 번째는 83.2%였다. 높은 점수를 받은 선택도 사진과 조건을 함께 확인할 필요가 있다는 사례다. 스크린샷에는 입력란과 선택 결과가 함께 담겨 있지만, 질문을 바꾼 뒤 매번 판단 버튼을 다시 눌렀는지는 화면만으로 확인할 수 없다. 세 번째의 흐릿한 D 사진의 중심 대상을 확정할 원본 기록도 없어, 그 경우의 정답은 여기서 단정하지 않겠다. 화면에 표시된 MPS 실행 시간은 1.37초, 1.09초, 1.12초이며 아래의 Windows GPU 비교와는 장비와 측정 맥락이 다르다.

'''
    marker='CPU/CUDA/학습용 구현을 별도 프로세스로 다시 불러온'
    # The editorial pass changes this paragraph; insert before it.
    marker='저장한 가중치를 다시 읽어도 같은 판단이 나올까?'
    assert marker in text
    text=text.replace(marker,snapshots+marker,1)
    text=text.replace('실제 Mac에서 동작과 속도를 검증하지 못했으며, 16 GB Mac에 충분히 들어간다고 보장하지 않는다.', '사용자가 제공한 위 화면에는 MPS 실행 사례가 보인다. 다만 실제 Mac의 기종별 메모리와 속도는 별도로 측정하지 않았으므로, 16 GB Mac에서 필요한 여유 공간은 직접 확인해야 한다.')
    return text
