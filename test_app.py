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
