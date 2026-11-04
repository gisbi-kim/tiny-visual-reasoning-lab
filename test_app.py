"""HTTP transport tests, explicitly using a stub rather than claiming model accuracy."""
import json
from pathlib import Path
import pytest
from app import create_app,ROOT
from schema import ACTIONS

@pytest.mark.skipif(not (ROOT/'data/demo.json').exists(),reason='Local Tiny ImageNet demo not downloaded')
def test_verbatim_input_and_seven_way_response():
    class Stub:
        device='test'
        def predict(self,image,text):
            self.received=(image.size,text)
            return {'action':'F','probabilities':{a:float(a=='F') for a in ACTIONS},'latency_ms':0,'device':'test'}
    engine=Stub();client=create_app(engine).test_client()
    assert client.get('/').status_code==200
    scene=client.post('/scene',json={'canonical':True}).get_json()
    instruction='  고양이에서 왼쪽, 다음 아래!  '
    response=client.post('/predict',json={**scene,'instruction':instruction})
    assert response.status_code==200
    assert engine.received==((384,256),instruction)
    assert response.get_json()['action']=='F'
    assert client.post('/predict',json={**scene,'instruction':''}).status_code==400
    assert client.post('/predict',json=[]).status_code==400


def test_scene_examples_use_board_objects_and_valid_moves(monkeypatch):
    import random
    import app as app_module
    from oracle import adjacent
    seen = []
    def capture(program, language, variant, seed):
        seen.append((program, language))
        return str(program)
    monkeypatch.setattr(app_module, 'instruction', capture)
    board = ['dog', 'banana', 'cat', 'lifeboat', 'goldfish', 'backpack']
    for seed in range(50):
        seen.clear()
        examples = app_module.scene_examples(board, random.Random(seed))
        assert len(examples) == 3
        branch, neighbor, chain = (entry[0] for entry in seen)
        assert branch['yes'] in board and branch['no'] in board and branch['yes'] != branch['no']
        for program in (neighbor, chain):
            start = board.index(program['anchor'])
            first = adjacent(start, program['direction'])
            assert first is not None
            if program['family'] == 'chain':
                assert adjacent(first, program['second_direction']) is not None
        assert seen[2][1] == 'en'

@pytest.mark.skipif(not (ROOT/'data/demo.json').exists(), reason='Local Tiny ImageNet demo not downloaded')
def test_new_scenes_return_three_usable_examples():
    class Stub:
        device = 'test'
    client = create_app(Stub()).test_client()
    for canonical in (True, False):
        response = client.post('/scene', json={'canonical': canonical})
        assert response.status_code == 200
        data = response.get_json()
        assert data['image'].startswith('data:image/png;base64,')
        assert len(data['examples']) == 3
        assert all(item['label'] and item['text'] for item in data['examples'])
