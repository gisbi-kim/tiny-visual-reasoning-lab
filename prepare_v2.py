"""All authored forms and all task families are TRAINED; hold out only source photos."""
from collections import Counter,defaultdict
import hashlib,json,random,re
from pathlib import Path
from language import instruction
from oracle import TRAIN_FAMILIES,HELDOUT_FAMILIES,execute
from prepare_data import ROOT,make_split,dump
from schema import ACTIONS

FAMILIES=TRAIN_FAMILIES+HELDOUT_FAMILIES
FORMS=tuple(range(6))

def surface(text,language,seed):
    """Deterministic meaning-preserving surface variations; no access to the answer."""
    rng=random.Random(seed^731203)
    if rng.random()<.5:
        if language=='ko':
            words={1:'한',2:'두',3:'세',4:'네',5:'다섯',6:'여섯'}
            text=re.sub(r'([1-6])장',lambda m:words[int(m[1])]+' 장',text)
        else:
            words={1:'one',2:'two',3:'three',4:'four',5:'five',6:'six'}
            text=re.sub(r'\b([1-6])\b',lambda m:words[int(m[1])],text)
    if language=='ko' and rng.random()<.5:
        for a,b in [('왼쪽','좌측'),('오른쪽','우측'),('위쪽으로','위로'),('아래쪽으로','아래로'),('위쪽','위'),('아래쪽','아래')]:text=text.replace(a,b)
    return text

def save(name,rows):
    (ROOT/'data'/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')

def reload_probe(selected):
    rows=[];rng=random.Random(85219)
    for family in FAMILIES:
        for language in ('ko','en'):
            action=ACTIONS[len(rows)%len(ACTIONS)]
            choices=[r for r in selected if r['family']==family and r['language']==language and r['action']==action]
            rows.append(rng.choice(choices))
    save('v2_reload_probe',rows)
    return rows

def main():
    if (ROOT/'data/v2_train.jsonl').exists():raise RuntimeError('Preserve existing v2 data; do not silently regenerate')
    pools=json.loads((ROOT/'data/source-partitions.json').read_text(encoding='utf-8'))
    ids={s:{r['id'] for rs in pool.values() for r in rs} for s,pool in pools.items()}
    hashes=[r['pixel_sha256'] for pool in pools.values() for rs in pool.values() for r in rs]
    assert len(hashes)==len(set(hashes))
    assert all(not ids[a]&ids[b] for a in ids for b in ids if a!=b)
    specs=[('v2_train','train',224),('v2_validation','validation',12),('v2_calibration','calibration',6),('v2_test','test',12)]
    splits={};summary={}
    for i,(name,source,quota) in enumerate(specs):
        rows,stats=make_split(name,pools[source],FAMILIES,quota,False,8507+i*101,form_choices=FORMS)
        coverage=Counter()
        for row in rows:
            row['instruction']=surface(row['instruction'],row['language'],row['text_seed'])
            row['surface_version']=2
            assert row['instruction']==surface(instruction(row['program'],row['language'],row['variant'],row['text_seed']),row['language'],row['text_seed'])
            assert set(row['source_ids'])<=ids[source]
            assert execute(row['object_keys'],row['program'])[0]==row['action']
            coverage[(row['family'],row['language'],row['variant'])]+=1
        assert all(coverage[(f,l,v)]>0 for f in FAMILIES for l in ('ko','en') for v in FORMS)
        stats['all_forms_coverage']={'|'.join(map(str,k)):v for k,v in sorted(coverage.items())}
        stats['unique_instructions']=len(set(r['instruction'] for r in rows));stats['source_partition']=source
        stats['forms']=FORMS;stats.pop('heldout_language',None)
        save(name,rows);splits[name]=rows;summary[name]=stats
        print('V2_AUDIT_OK',name,len(rows),len(coverage),flush=True)
    # Two rows per family/language/action; greedily spread all six forms.
    groups=defaultdict(list)
    for r in splits['v2_validation']:groups[(r['family'],r['language'],r['action'])].append(r)
    selected=[];used=Counter()
    for f in FAMILIES:
        for lang in ('ko','en'):
            for action in ACTIONS:
                candidates=groups[(f,lang,action)]
                chosen=[]
                for _ in range(2):
                    row=min((r for r in candidates if r not in chosen),key=lambda r:used[(f,lang,r['variant'])])
                    chosen.append(row);selected.append(row);used[(f,lang,row['variant'])]+=1
            assert all(used[(f,lang,v)]>0 for v in FORMS)
    save('v2_validation_select',selected)
    probe=reload_probe(selected)
    assert len(selected)==336
    summary['selection_rows']=len(selected)
    summary['reload_probe_rows']=len(probe)
    summary['policy']='All 12 families and all six forms per language appear in every split. Split by source photos, NOT by wording or task composition. Original v1 test source pool reused; this is a practical known-task evaluation, not a blind novel-task claim.'
    summary['audits']={'all_144_authored_forms_in_train':True,'all_12_families_in_train':True,'source_ids_disjoint':True,'source_rgb_hashes_disjoint':True,'all_answers_recomputed':True,'all_texts_reproduced':True,'no_v1_test_photos_in_training':True}
    summary['files']={n:hashlib.sha256((ROOT/'data'/f'{n}.jsonl').read_bytes()).hexdigest() for n in [*splits,'v2_validation_select','v2_reload_probe']}
    dump(ROOT/'docs/v2-data-summary.json',summary)
    print('V2_DATA_READY',flush=True)

if __name__=='__main__':main()
