import random
from catalog import CATALOG
from oracle import execute,conditional_can_vary,TRAIN_FAMILIES,HELDOUT_FAMILIES
from prepare_data import proposal,counterfactual
from language import instruction,KO,EN
from schema import model_input

BOARD=['dog','banana','cat','lifeboat','goldfish','backpack']
def test_known_programs():
    cases=[({'family':'direct','target':'cat'},'C'),
        ({'family':'neighbor','anchor':'cat','direction':'down'},'F'),
        ({'family':'row_filter','anchor':'dog','group':'food'},'B'),
        ({'family':'col_filter','anchor':'cat','group':'animal'},'C'),
        ({'family':'row_except','anchor':'dog','group':'animal'},'B'),
        ({'family':'neighbor_row_filter','anchor':'cat','direction':'down','group':'fish'},'E'),
        ({'family':'neighbor_col_filter','anchor':'cat','direction':'left','group':'fish'},'E'),
        ({'family':'row_filter_then_neighbor','anchor':'dog','group':'food','direction':'down'},'E'),
        ({'family':'chain','anchor':'cat','direction':'left','second_direction':'down'},'E'),
        ({'family':'count_branch','group':'fruit','threshold':2,'yes':'goldfish','no':'dog'},'A'),
        ({'family':'compare_count','group':'animal','other_group':'food','yes':'backpack','no':'cat'},'F'),
        ({'family':'neighbor_filter','anchor':'goldfish','direction':'right','group':'animal'},'NONE')]
    for p,expected in cases:assert execute(BOARD,p)[0]==expected
def test_ambiguity_and_edges():
    assert execute(BOARD,{'family':'direct','target':'animal'})[0] is None
    assert execute(BOARD,{'family':'neighbor','anchor':'dog','direction':'up'})[0]=='NONE'
    assert execute(BOARD,{'family':'neighbor','anchor':'cat','direction':'right'})[0]=='NONE'
    assert execute(BOARD,{'family':'row_filter','anchor':'elephant','group':'animal'})[0]=='NONE'
def test_horizontal_reflection():
    rng=random.Random(123);mapping=dict(zip('ABCDEF','CBAFED'));mapping.update(NONE='NONE');mapping[None]=None
    for family in TRAIN_FAMILIES+HELDOUT_FAMILIES:
        for _ in range(100):
            board=rng.sample(list(CATALOG),6);p=proposal(board,family,rng);q=dict(p)
            for field in ('direction','second_direction'):
                if field in q:q[field]={'left':'right','right':'left','up':'up','down':'down'}[q[field]]
            flipped=[board[i] for i in (2,1,0,5,4,3)]
            assert execute(flipped,q)[0]==mapping[execute(board,p)[0]]
def test_all_authored_forms_render():
    rng=random.Random(9)
    for family in TRAIN_FAMILIES+HELDOUT_FAMILIES:
        p=proposal(BOARD,family,rng)
        assert len(KO[family])==len(EN[family])==6
        for lang in ('ko','en'):
            for variant in range(6):
                text=instruction(p,lang,variant,42)
                assert text and '{' not in text
                assert text==instruction(p,lang,variant,42)
def test_input_boundary():
    text='고양이 아래쪽 사진을 골라.';payload=model_input('pixels',text)
    assert payload['state']==text
    assert set(payload)=={'state','images','questions'}
    assert not any(k in payload for k in ('program','trace','object_keys','action','source_ids'))

def test_conditional_difficulty_uses_domain_constraints():
    assert not conditional_can_vary({'family':'count_branch','group':'boat','threshold':3})
    assert conditional_can_vary({'family':'count_branch','group':'boat','threshold':1})
    assert not conditional_can_vary({'family':'compare_count','group':'mammal','other_group':'animal'})
    assert conditional_can_vary({'family':'compare_count','group':'animal','other_group':'mammal'})
