"""Build audited bilingual multi-step boards from original Tiny ImageNet IDs."""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import hashlib,json,random
from pathlib import Path
import numpy as np
from PIL import Image
from catalog import CATALOG,GROUPS,matches
from oracle import TRAIN_FAMILIES,HELDOUT_FAMILIES,DIRECTIONS,execute
from language import instruction
from render import render,SIZE
from schema import ACTIONS
ROOT=Path(__file__).resolve().parent
RAW=ROOT/'data/raw/tiny-imagenet-200'
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def build_index():
    lookup={v['wnid']:k for k,v in CATALOG.items()}
    available=set((RAW/'wnids.txt').read_text().splitlines())
    assert set(lookup)<=available
    groups={s:defaultdict(list) for s in ('train','validation','calibration','test')}
    rejected=Counter();hashes=set();rng=random.Random(3407)
    def item(path,box,key):
        x0,y0,x1,y1=map(int,box)
        x0=max(0,x0);y0=max(0,y0);x1=min(64,x1+1);y1=min(64,y1+1)
        if min(x1-x0,y1-y0)<20 or (x1-x0)*(y1-y0)<512:
            rejected['small_annotated_object']+=1;return
        with Image.open(RAW/path) as im:
            if im.size!=(64,64):rejected['wrong_size']+=1;return
            digest=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
        if digest in hashes:rejected['exact_pixel_duplicate']+=1;return
        hashes.add(digest)
        return {'id':path,'key':key,'box':[x0,y0,x1,y1],'pixel_sha256':digest}
    for wnid,key in lookup.items():
        rows=[]
        for line in (RAW/f'train/{wnid}/{wnid}_boxes.txt').read_text().splitlines():
            name,*box=line.split();row=item(f'train/{wnid}/images/{name}',box,key)
            if row:rows.append(row)
        rng.shuffle(rows)
        assert len(rows)>160,(key,len(rows))
        groups['validation'][key]=rows[:40];groups['calibration'][key]=rows[40:60];groups['train'][key]=rows[60:]
    for line in (RAW/'val/val_annotations.txt').read_text().splitlines():
        name,wnid,*box=line.split()
        if wnid in lookup:
            row=item(f'val/images/{name}',box,lookup[wnid])
            if row:groups['test'][lookup[wnid]].append(row)
    assert all(len(groups[s][k])>=5 for s in groups for k in CATALOG)
    dump(ROOT/'data/source-partitions.json',groups)
    dump(ROOT/'docs/source-quality.json',{'criteria':'Before model training: original RGB pixel duplicates removed globally; annotated object width/height >=20 and area >=512 pixels. Inclusive source boxes converted to exclusive PIL boxes. Full-image pixel hashes prevent exact duplicate leakage.',
        'rejected':rejected,'counts':{s:{k:len(v) for k,v in g.items()} for s,g in groups.items()},
        'official_unlabeled_test':'Not used. The labeled official validation set is our final test source.',
        'limitations':'Bounding-box/class labels do not guarantee a photograph is visually unambiguous; near-duplicates and base-model pretraining exposure are not audited.'})
    return groups
def crop(item):
    with Image.open(RAW/item['id']) as image:return image.convert('RGB').crop(tuple(item['box']))
def proposal(board,family,rng):
    keys=list(CATALOG);absent=[k for k in keys if k not in board]
    p={'family':family}
    def subject():return rng.choice(board if rng.random()<.88 else absent)
    if family=='direct':p['target']=rng.choice(board if rng.random()<.8 else absent)
    elif family in ('count_branch','compare_count'):
        p['group']=rng.choice(list(GROUPS))
        p['yes']=subject();p['no']=subject()
        while p['no']==p['yes']:p['no']=subject()
        if family=='count_branch':p['threshold']=rng.choice((1,2,3))
        else:
            p['other_group']=rng.choice([g for g in GROUPS if g!=p['group']])
    else:
        p['anchor']=subject()
        if family in ('neighbor','neighbor_filter','chain','neighbor_row_filter','neighbor_col_filter','row_filter_then_neighbor'):
            p['direction']=rng.choice(DIRECTIONS)
        if family=='chain':
            # Only perpendicular paths: both moves matter, with no canceling loop.
            p['second_direction']=rng.choice(('up','down') if p['direction'] in ('left','right') else ('left','right'))
        if family in ('row_filter','col_filter','row_except','neighbor_filter','neighbor_row_filter','neighbor_col_filter','row_filter_then_neighbor'):
            p['group']=rng.choice(list(GROUPS))
    return p
def counterfactual(p,board):
    q=deepcopy(p);f=p['family']
    if 'direction' in p:q['direction']={'left':'right','right':'left','up':'down','down':'up'}[p['direction']]
    elif f=='count_branch':
        count=sum(matches(k,p['group']) for k in board)
        q['threshold']=count+1 if count>=p['threshold'] else max(1,count)
        if q['threshold']>6 or q['threshold']==p['threshold']:return None
    elif f=='compare_count':q['group'],q['other_group']=q['other_group'],q['group']
    elif f=='direct':
        q['target']=next(k for k in board if k!=p['target'])
    else:return None
    return q
def make_split(name,pool,families,quota,heldout_language,seed,form_choices=None):
    rng=random.Random(seed);counts=Counter();rows=[];scene_rows=defaultdict(list);pairs=0;scenes=0
    keys=[]
    for f in families:
        for lang in ('ko','en'):
            for action in ACTIONS:
                for truth in ((False,True) if f in ('count_branch','compare_count') else (None,)):
                    keys.append((f,lang,action,truth))
    limits={k:(quota//2 if k[3] is not None else quota) for k in keys}
    remaining=sum(limits.values());used=set();digit_counts=Counter();group_counts=Counter();slot_counts=Counter();programs=Counter();variants=Counter()
    def record_key(program,lang,action,trace):
        truth=trace[0]['condition'] if program['family'] in ('count_branch','compare_count') else None
        return (program['family'],lang,action,truth)
    attempt=0
    while remaining:
        attempt+=1
        if attempt>250000:raise RuntimeError(('Quota generation exhausted',name,{str(k):limits[k]-counts[k] for k in limits if limits[k]!=counts[k]}))
        board=rng.sample(list(CATALOG),6);accepted=[];seen=set()
        active=[f for f in families if any(counts[k]<limits[k] for k in keys if k[0]==f)]
        for _ in range(90):
            if len(accepted)>=8:break
            p=proposal(board,rng.choice(active),rng)
            candidates=[p]
            q=counterfactual(p,board)
            if q and execute(board,q)[0]!=execute(board,p)[0]:candidates.append(q)
            pair_id=f'{name}-pair-{attempt}-{_}' if len(candidates)==2 else None
            variant=rng.choice(form_choices) if form_choices is not None else rng.randrange(4,6) if heldout_language else rng.randrange(4)
            text_seed=rng.randrange(2**31)
            for candidate in candidates:
                signature=json.dumps(candidate,sort_keys=True)
                if signature in seen or len(accepted)>=8:continue
                seen.add(signature)
                answer,trace=execute(board,candidate)
                if answer is None:continue
                language_rows=[]
                for lang in ('ko','en'):
                    key=record_key(candidate,lang,answer,trace)
                    if counts[key]>=limits[key]:continue
                    text=instruction(candidate,lang,variant,text_seed)
                    language_rows.append(dict(instruction=text,action=answer,language=lang,family=candidate['family'],
                        program=candidate,trace=trace,variant=variant,text_seed=text_seed,contrast_pair=pair_id))
                    counts[key]+=1;remaining-=1;variants[f'{lang}:{variant}']+=1
                accepted.extend(language_rows)
        if not accepted:continue
        scene_id=f'{name}-{scenes:05d}';scenes+=1
        items=[rng.choice(pool[k]) for k in board]
        seed_value=rng.randrange(2**31)
        image_path=f'images/{scene_id}.png'
        output=ROOT/'data'/image_path;output.parent.mkdir(exist_ok=True)
        render([crop(item) for item in items],seed_value).save(output)
        for item in items:used.add(item['id']);digit_counts[item['key']]+=1
        for slot,key in enumerate(board):
            slot_counts[f'{slot}:{key}']+=1
            for group in CATALOG[key]['groups']:group_counts[group]+=1
        for row in accepted:
            row.update(scene_id=scene_id,image=image_path,source_ids=[i['id'] for i in items],object_keys=board,
                       boxes=[i['box'] for i in items],render_seed=seed_value)
            rows.append(row);scene_rows[scene_id].append(row);programs[row['family']]+=1
        if scenes%500==0:print('BUILD',name,'scenes',scenes,'rows',len(rows),'remaining',remaining,flush=True)
    rng.shuffle(rows)
    (ROOT/'data'/f'{name}.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    paired=defaultdict(list)
    for r in rows:
        if r['contrast_pair']:paired[(r['contrast_pair'],r['language'])].append(r)
    meaningful_pairs=sum(len(set(r['action'] for r in g))>=2 for g in paired.values())
    summary={'rows':len(rows),'scenes':scenes,'unique_source_images':len(used),'unique_instructions':len(set(r['instruction'] for r in rows)),
        'actions':dict(Counter(r['action'] for r in rows)),'languages':dict(Counter(r['language'] for r in rows)),
        'families':programs,'quotas':{'|'.join(map(str,k)):v for k,v in counts.items()},'source_object_occurrences':digit_counts,
        'source_group_occurrences':group_counts,'source_objects_by_slot':slot_counts,'language_variants':variants,
        'different_answer_scenes':sum(len(set(r['action'] for r in g))>=2 for g in scene_rows.values()),
        'different_answer_contrast_pairs_by_language':meaningful_pairs,'seed':seed,'heldout_language':heldout_language}
    print('BUILT',name,len(rows),scenes,flush=True)
    return rows,summary
def audit(splits,partitions):
    pools={name:{r['id'] for items in pool.values() for r in items} for name,pool in partitions.items()}
    for a in pools:
        for b in pools:
            if a!=b:assert not pools[a]&pools[b]
    all_hashes=[r['pixel_sha256'] for pool in partitions.values() for items in pool.values() for r in items]
    assert len(all_hashes)==len(set(all_hashes))
    for name,rows in splits.items():
        source='test' if name.startswith('test') else name
        for row in rows:
            assert set(row['source_ids'])<=pools[source]
            assert execute(row['object_keys'],row['program'])[0]==row['action']
            assert row['instruction']==instruction(row['program'],row['language'],row['variant'],row['text_seed'])
            assert (row['variant']>=4)==(name in ('test_language','test_combined'))
            assert (row['family'] in HELDOUT_FAMILIES)==(name in ('test_composition','test_combined'))
        print('AUDIT_OK',name,len(rows),flush=True)
    return {'source_id_disjoint':True,'exact_rgb_hash_disjoint':True,'oracle_recomputed':True,'text_reproduced':True,'heldout_forms_excluded':True,'heldout_program_families_excluded':True}
def examples(partitions):
    pictures=[];labels=[];ids=[]
    for key,items in partitions['test'].items():
        for item in items[:10]:
            pictures.append(np.array(crop(item).resize((64,64),Image.Resampling.BICUBIC)))
            labels.append(key);ids.append(item['id'])
    np.savez_compressed(ROOT/'data/examples.npz',images=np.array(pictures),labels=np.array(labels),source_ids=np.array(ids))
def main():
    p=argparse.ArgumentParser();p.add_argument('--examples-only',action='store_true');args=p.parse_args()
    if (ROOT/'data/train.jsonl').exists() and not args.examples_only:raise RuntimeError('Preserve existing dataset; use a separate project/data root for regeneration')
    partitions=build_index();examples(partitions)
    if args.examples_only:return
    specifications=[('train','train',TRAIN_FAMILIES,224,False),('validation','validation',TRAIN_FAMILIES,8,False),
        ('calibration','calibration',TRAIN_FAMILIES,4,False),('test_id','test',TRAIN_FAMILIES,8,False),
        ('test_language','test',TRAIN_FAMILIES,8,True),('test_composition','test',HELDOUT_FAMILIES,40,False),
        ('test_combined','test',HELDOUT_FAMILIES,40,True)]
    splits={};summary={}
    for i,(name,source,families,quota,heldout) in enumerate(specifications):
        splits[name],summary[name]=make_split(name,partitions[source],families,quota,heldout,3407+i*101)
    summary['audit']=audit(splits,partitions)
    summary['renderer']={'size':SIZE,'tile_size':128,'max_photo_size':112,'resize':'BICUBIC, preserve aspect ratio','color_space':'RGB','source':'annotated bounding-box crop'}
    dump(ROOT/'docs/data-summary.json',summary)
    # A fixed, balanced validation subset: one row per family/language/action.
    chosen={}
    for row in splits['validation']:chosen.setdefault((row['family'],row['language'],row['action']),row)
    (ROOT/'data/validation_select.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen.values()),encoding='utf-8')
    dump(ROOT/'docs/catalog.json',{'objects':CATALOG,'groups':GROUPS})
    print('DATA_READY',flush=True)
if __name__=='__main__':main()
