"""Compose the four paired generative runs from their saved measurements."""
import json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/generative-comparison'

def create():
    read=lambda name:json.loads((OUT/name).read_text(encoding='utf-8'))
    decision=read('decision.json'); old=read('generative.json')
    configs=[(False,40),(True,40),(False,120),(True,120)]
    runs=[read(f'thinking-{int(t)}-tokens-{n}.json') for t,n in configs]
    for run in runs:
        assert run['count']==96
        for row,base in zip(run['rows'],decision['rows']):
            assert row['scene_id']==base['scene_id'] and row['instruction']==base['instruction'] and row['gold']==base['gold']
    def pct(n):return f'{n/96*100:.2f}%'
    def old_answer(row):
        if row['action']: return row['action']
        match=re.fullmatch(r'\s*(?:정답|답|answer)\s*[:：]\s*(NONE|[A-F])\s*[.!]?\s*',row['raw'],re.I)
        return match.group(1).upper() if match else None
    old_relaxed=sum(old_answer(row)==row['gold'] for row in old['rows'])
    table=['| 모델·설정 | 엄격 정답 | 형식 허용 정답 | 최종 답 미인식 | 한도 도달 | 중앙 시간 |',
           '|---|---:|---:|---:|---:|---:|',
           f"| decision model v2 | {decision['correct']}/96 ({pct(decision['correct'])}) | 같은 값 | 0 | 해당 없음 | {decision['median_ms']:.1f}ms |",
           f"| 생성형 · 생각 끔 · 8토큰 (기존) | {old['correct']}/96 ({pct(old['correct'])}) | {old_relaxed}/96 ({pct(old_relaxed)}) | {old['invalid']} | 기록 없음 | {old['median_ms']:.1f}ms |"]
    for run in runs:
        mode='켬' if run['thinking'] else '끔'
        table.append(f"| 생성형 · 생각 {mode} · {run['budget']}토큰 | {run['strict_correct']}/96 ({pct(run['strict_correct'])}) | {run['relaxed_correct']}/96 ({pct(run['relaxed_correct'])}) | {run['invalid']} | {run['hit_token_limit']} | {run['median_ms']:.1f}ms |")
    links=' · '.join(f"[생각 {'켬' if run['thinking'] else '끔'} · {run['budget']}토큰](../results/generative-comparison/{run['config']}.json)" for run in runs)
    details=', '.join(f"생각 {'켬' if run['thinking'] else '끔'}·{run['budget']}토큰 {run['no_final']}개" for run in runs)
    summary={'count':96,'decision_correct':decision['correct'],'decision_median_ms':decision['median_ms'],
             'old_8_token_strict_correct':old['correct'], 'old_8_token_relaxed_correct':old_relaxed,
             'runs':[{k:run[k] for k in ('config','budget','thinking','strict_correct','relaxed_correct','no_final','invalid','hit_token_limit','median_ms','median_generated_tokens')} for run in runs],
             'evaluation':'identical seed-3407 96-example sample, greedy decoding, same prompt, 4bit/BF16, synchronized CUDA, model load excluded',
             'final_answer_parsing':'Only text following </think> when thinking is enabled; strict A-F/NONE or relaxed prefix answer:/정답:. Incomplete or unparsable counts wrong.',
             'limitations':'base generative model vs fine-tuned decision model; prompt and decoding budgets affect results; 96 rows share source images.'}
    (OUT/'reasoning-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return f'''## 생성형 Qwen에게 생각할 시간을 더 주면?

앞에서 선택 점수를 계산하는 방식과 글자를 직접 생성하는 방식을 비교했다. 처음 비교에서는 생성형에 답 한 글자만 요청하고 최대 8토큰을 주었다. 그런데 출력 중 설명을 시작하다가 답을 끝내지 못한 경우가 있었다. 그렇다면 토큰을 더 주거나, 생각 모드를 켜면 결과가 어떻게 달라질까?

같은 `v2_test`의 **96문제**로 다시 시험했다. 12개 작업 유형과 한국어·영어에서 각각 네 문제씩 고른 표본이며, 사진 파일 해시와 질문·정답이 앞선 비교와 일치한다. 생성형은 파인튜닝하지 않은 Qwen3.5-4B, decision model은 같은 기반 모델에서 추가 학습한 v2이다. 모두 RTX 4070 Ti SUPER에서 NF4 4bit/BF16으로 실행했다.

v2는 여기서 평가하는 작업 유형과 문장 틀을 학습했다. 평가에 쓰인 사진은 v2 학습에서 제외했다. 같은 사진판이 여러 문제에 쓰이는 96문제이므로, 서로 독립적인 사진 96장에 대한 결과로 해석하지 않는다.

생성형에게 매번 같은 질문과 A–F 또는 NONE으로 답하라는 지시를 주고, 토큰 한도를 40과 120으로 바꾸었다. 각각 Qwen의 생각 모드를 끈 경우와 켠 경우를 실행했으며, 네 조건 모두 무작위 샘플링 없이 결정적으로 생성했다.

{chr(10).join(table)}

먼저 생각을 끈 두 조건을 보면, 형식을 조금 허용했을 때 40토큰에서 {runs[0]['relaxed_correct']}개, 120토큰에서 {runs[2]['relaxed_correct']}개를 맞혔다. 생각을 켠 경우에는 40토큰에서 {96-runs[1]['no_final']}개, 120토큰에서 {96-runs[3]['no_final']}개가 생각 구간을 닫고 답변 단계에 도달했다. 한도 안에 답이 나오지 않은 문제는 점수 계산에서 오답이 되므로, 이 숫자와 정답 수를 함께 봐야 한다. 특히 여기의 0/96은 **한도 안에 채점할 최종 답을 얻지 못했다**는 뜻이다. 시간을 더 줬을 때 생각의 결론이 맞을지 틀릴지는 이 결과만으로 알 수 없다.

엄격 정답은 최종 출력이 A–F 또는 NONE 하나인 경우만 인정한다. 형식 허용 정답은 `정답: B`, `Answer: B`도 읽는다. 최종 답이 형식에 맞지 않거나 끝내 나오지 않으면 정답 수에서는 오답으로 계산했다. 한도 도달은 EOS 종료 토큰을 내기 전에 최대 생성 토큰 수에 이른 경우다. 표의 미인식과 한도 도달은 겹칠 수 있다.

생각 모드의 출력은 `</think>` 뒤에 나온 최종 답만 채점했다. 생각 중간에 쓰인 A나 B는 답으로 세지 않았다. 생각 구간을 닫지 못해 최종 답변을 찾지 못한 수는 {details}다. 생각 끔은 답변 단계에서 시작한다.

시간은 세 문제의 준비 실행 뒤, 이미지를 연 상태에서 전처리·모델 계산·결과 변환을 포함해 GPU 완료까지 잰 중앙값이다. 모델 로딩과 파일 읽기·HTTP 시간은 빠져 있다. 기존 8토큰과 decision 값은 앞선 별도 실행에서 잰 값이며 네 새 조건은 이후 같은 컴퓨터에서 측정했다. 긴 출력을 생성하는 조건에서는 그 토큰을 생성하는 시간도 들어간다.

이 표는 **기본 생성형 모델**과 **LoRA·head를 추가 학습한 모델**의 실제 경로를 비교한다. 따라서 정확도 차이에는 추가 학습과 출력 방식이 함께 작용한다. 또 생각 모드에 준 40·120토큰은 최종 답에 도달하기 전의 생각에도 사용된다. 다른 프롬프트나 더 긴 한도에서는 결과가 달라질 수 있으므로, 각 조건의 원문과 종료 상태를 함께 살펴보자.

{links} · [비교 요약](../results/generative-comparison/reasoning-summary.json) · [평가 코드](../compare_reasoning_modes.py) · [기존 8토큰 출력](../results/generative-comparison/generative.json).

'''
