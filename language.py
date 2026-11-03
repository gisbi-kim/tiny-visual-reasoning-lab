"""Authored semantic paraphrases: train forms 0..3; held-out structures 4..5.

Text is rendered from a program, never from its answer. Source image IDs, slot
answers and oracle traces are not available to this module.
"""
import random
from catalog import CATALOG,GROUPS
KO={
'direct':[
'{t} 사진을 골라.', '{t}이 보이는 칸을 선택해 줘.', '사진들 중에서 {t}을 찾아줘.', '{t} 사진은 어디 있어?',
'선택할 대상은 {t}이야. 해당 사진의 위치로 답해.', '어느 칸을 눌러야 {t} 사진을 고르게 될까?'],
'neighbor':[
'{a} 사진에서 {d}으로 한 칸 이동한 사진을 골라.', '{a} 사진의 바로 {n} 사진을 선택해.', '{a}을 찾은 다음, 그 칸에서 {d}으로 한 칸 가 줘.', '{a} 사진과 {d}으로 바로 이웃한 사진은 어느 칸이야?',
'출발점을 {a} 사진으로 잡아. {d}으로 딱 한 칸 간 곳이 목적지야.', '답으로 고를 사진의 바로 {opn}에 {a} 사진이 있어야 해. 그 사진을 찾아.'],
'row_filter':[
'{a} 사진과 같은 행에 있는 {g} 사진을 골라.', '{a}이 있는 가로줄에서 {g} 사진을 찾아줘.', '먼저 {a} 사진의 행을 찾고, 그 행의 {g} 사진을 선택해.', '{g} 사진 중 {a} 사진과 가로줄이 같은 것은 어디 있어?',
'고를 사진은 {g} 사진이면서 {a} 사진과 행이 같아야 해.', '가로줄을 {a} 사진이 있는 쪽으로 한정하면, {g} 사진의 위치는?'],
'col_filter':[
'{a} 사진과 같은 열에 있는 {g} 사진을 골라.', '{a}이 있는 세로줄에서 {g} 사진을 찾아줘.', '먼저 {a} 사진의 열을 찾고, 그 열의 {g} 사진을 선택해.', '{g} 사진 중 {a} 사진과 세로줄이 같은 것은 어디 있어?',
'고를 사진은 {g} 사진이면서 {a} 사진과 열이 같아야 해.', '세로줄을 {a} 사진이 있는 쪽으로 한정하면, {g} 사진의 위치는?'],
'neighbor_filter':[
'{a} 사진의 바로 {n}에 있는 {g} 사진을 골라.', '{a}에서 {d}으로 한 칸 간 곳이 {g} 사진이면 그 칸을 선택해.', '{a}의 {d} 이웃 칸을 확인해서 {g} 사진일 때만 골라줘.', '{g} 사진이 {a} 사진의 바로 {n}에 있으면 선택해.',
'{g} 사진을 찾되, 그 사진의 바로 {opn}에 {a} 사진이 있어야 해.', '목적지는 {a}에서 {d} 한 칸이야. 목적지가 {g} 사진일 때만 답하고, 아니면 NONE이야.'],
'chain':[
'{a} 사진에서 {d}으로 한 칸, 이어서 {e}으로 한 칸 이동한 사진을 골라.', '{a}을 찾고 {d} 한 칸 간 다음 {e} 한 칸 가 줘.', '{a}의 바로 {n} 사진에서 다시 {e}으로 한 칸 이동해. 도착한 칸을 선택해.', '순서대로 {a} 찾기, {d} 한 칸, {e} 한 칸. 마지막 사진은 어디야?',
'시작은 {a} 사진이고 이동 순서는 {d}, {e}이야. 각 이동은 한 칸씩이야. 마지막 칸으로 답해.', '첫 이동은 {d} 한 칸, 두 번째는 {e} 한 칸이야. {a}에서 출발했을 때 도착하는 사진을 찾아.'],
'count_branch':[
'{g} 사진이 {k}장 이상이면 {y} 사진을, 아니면 {z} 사진을 골라.', '{g} 사진 수를 세어 봐. {k}장 이상이면 {y}, 부족하면 {z}을 선택해.', '{k}장 이상의 {g} 사진이 있으면 {y}을 찾아줘. 그렇지 않으면 {z}을 찾아줘.', '{g} 사진 개수가 {k} 이상일 때는 {y}, 그 외에는 {z}. 해당 사진은 어디야?',
'기본 선택은 {z} 사진이지만, {g} 사진이 적어도 {k}장이면 {y} 사진으로 바꿔.', '{g} 사진이 {k}장 미만인 경우에는 {z} 사진으로 답해. 나머지 경우에는 {y} 사진으로 답해.'],
'compare_count':[
'{g} 사진이 {h} 사진보다 많으면 {y}, 아니면 {z} 사진을 골라.', '{g} 사진 수와 {h} 사진 수를 비교해. 앞쪽이 더 많을 때만 {y}, 나머지는 {z}을 선택해.', '{g} 사진 개수가 {h} 사진 개수보다 클 때 {y} 사진을 선택하고, 같거나 작으면 {z} 사진을 선택해.', '{g} 사진이 더 많아, {h} 사진이 더 많아? {g} 쪽이 더 많으면 {y}, 동점도 포함해서 그 외에는 {z}을 골라.',
'기본 답은 {z} 사진이야. {g} 사진 수가 {h} 사진 수를 넘는 경우에만 {y} 사진으로 바꿔.', '{h} 사진 수가 {g} 사진 수 이상이면 {z} 사진을 골라. 그렇지 않을 때는 {y} 사진이야.'],
'row_except':[
'{a} 사진과 같은 행에서 {g} 사진을 제외하고 남는 사진을 골라.', '{a}이 있는 가로줄에서 {g}에 해당하지 않는 사진을 선택해.', '{a}의 행을 찾고 그 행의 {g} 사진들을 빼 줘. 남은 사진은 어디야?', '{g} 사진이 아니면서 {a} 사진과 같은 행에 있는 사진을 찾아.',
'선택 범위는 {a} 사진의 가로줄이야. 그 안에서 {g} 사진은 선택하면 안 돼. 남은 칸으로 답해.', '{a} 사진이 있는 행만 남기고 그중 {g} 사진도 지운다고 생각해 봐. 남는 사진을 골라.'],
'neighbor_row_filter':[
'{a} 사진의 바로 {n} 칸과 같은 행에 있는 {g} 사진을 골라.', '{a}에서 {d}으로 한 칸 간 다음, 도착한 행에서 {g} 사진을 찾아.', '{a}의 {d} 이웃 칸을 기준으로 같은 가로줄의 {g} 사진을 선택해.', '먼저 {a}을 찾고 {d} 한 칸 이동해. 그 칸과 행이 같은 {g} 사진은 어디야?',
'{g} 사진을 고르되, {a}에서 {d} 한 칸 이동해서 도착한 행 안에서 찾아야 해.', '검색할 가로줄은 {a} 사진의 바로 {n} 칸이 있는 줄이야. 그 줄에서 {g} 사진으로 답해.'],
'neighbor_col_filter':[
'{a} 사진의 바로 {n} 칸과 같은 열에 있는 {g} 사진을 골라.', '{a}에서 {d}으로 한 칸 간 다음, 도착한 열에서 {g} 사진을 찾아.', '{a}의 {d} 이웃 칸을 기준으로 같은 세로줄의 {g} 사진을 선택해.', '먼저 {a}을 찾고 {d} 한 칸 이동해. 그 칸과 열이 같은 {g} 사진은 어디야?',
'{g} 사진을 고르되, {a}에서 {d} 한 칸 이동해서 도착한 열 안에서 찾아야 해.', '검색할 세로줄은 {a} 사진의 바로 {n} 칸이 있는 줄이야. 그 줄에서 {g} 사진으로 답해.'],
'row_filter_then_neighbor':[
'{a} 사진과 같은 행의 {g} 사진을 찾고, 그 사진에서 {d}으로 한 칸 이동한 사진을 골라.', '{a}이 있는 가로줄에서 {g} 사진을 찾은 뒤 그 바로 {n} 사진을 선택해.', '먼저 {a}의 행에 있는 {g} 사진을 찾아. 그 칸의 {d} 이웃을 골라줘.', '{a}과 같은 행의 {g} 사진에서 {d} 한 칸 떨어진 사진은 어디야?',
'이동을 시작할 칸은 {a} 사진과 같은 행에 있는 {g} 사진이야. 거기서 {d}으로 한 칸 이동해.', '마지막에 고를 사진은 {g} 사진의 바로 {n} 칸이야. 그 {g} 사진은 {a}과 같은 행에서 찾아.'],
}
EN={
'direct':['Choose the {t} photo.','Select the tile showing {t}.','Find the picture of {t}.','Which tile shows {t}?','The requested subject is {t}; give its tile.','Which tile should I click to select the {t} picture?'],
'neighbor':['Choose the photo one tile {d} of the {a} photo.','Select the immediate {n} neighbor of the {a} picture.','Find {a}, then move one tile {d}.','Which photo is immediately {d} of {a}?','Start at {a}. Your destination is exactly one tile {d}. Select it.','Find the photo whose immediate {opn} neighbor shows {a}.'],
'row_filter':['Choose the {g} photo in the same row as {a}.','Find {g} in the horizontal row containing {a}.','Locate the row of {a}, then select its {g} picture.','Which {g} picture shares a row with {a}?','The selected picture must show {g} and share a row with {a}.','Restrict your search to the row containing {a}; where is the {g} picture?'],
'col_filter':['Choose the {g} photo in the same column as {a}.','Find {g} in the vertical column containing {a}.','Locate the column of {a}, then select its {g} picture.','Which {g} picture shares a column with {a}?','The selected picture must show {g} and share a column with {a}.','Restrict your search to the column containing {a}; where is the {g} picture?'],
'neighbor_filter':['Choose the {g} photo immediately {d} of {a}.','Move one tile {d} from {a}; select that tile only if it shows {g}.','Check the immediate {n} neighbor of {a}. Choose it only if it is a {g} picture.','If a {g} photo is immediately {d} of {a}, select it.','Find a {g} picture whose immediate {opn} neighbor shows {a}.','Your destination is one tile {d} from {a}. Answer with that tile if it shows {g}, otherwise NONE.'],
'chain':['From {a}, move one tile {d}, then one tile {e}. Select the destination.','Find {a}, go one tile {d}, and then one tile {e}.','Start at the immediate {n} neighbor of {a} and move one tile {e}. Choose the final photo.','In order: find {a}, step {d}, step {e}. Each step is one tile. Where do you end?','Starting at {a}, follow these one-tile directions: {d}, {e}. Give the final tile.','The first move is one tile {d}; the second is one tile {e}. What photo do you reach from {a}?'],
'count_branch':['If there are at least {k} {g} photos, choose {y}; otherwise choose {z}.','Count the {g} pictures. Select {y} if the count is at least {k}, or {z} if it is lower.','With {k} or more {g} photos present, pick {y}. In all other cases, pick {z}.','Is the number of {g} photos at least {k}? If yes select {y}; if not select {z}.','The default choice is {z}; switch to {y} only when at least {k} tiles show {g}.','When fewer than {k} pictures show {g}, answer with {z}. Otherwise answer with {y}.'],
'compare_count':['If {g} photos outnumber {h} photos, choose {y}; otherwise choose {z}.','Compare the counts of {g} and {h} pictures. Choose {y} only if the first is larger; otherwise {z}.','Select {y} when there are more {g} photos than {h} photos. If the counts are equal or the first is smaller, select {z}.','Are there more {g} than {h} pictures? If yes choose {y}; if not, including ties, choose {z}.','The default answer is {z}. Change it to {y} only if the {g} count exceeds the {h} count.','If the number of {h} pictures is at least the number of {g} pictures, choose {z}. In the remaining case, choose {y}.'],
'row_except':['In the row containing {a}, exclude {g} pictures and choose the remaining photo.','Choose the photo that is not {g} in the same row as {a}.','Find the row of {a}; remove its {g} pictures. Which photo remains?','Find a non-{g} photo that shares a row with {a}.','Only search the horizontal row containing {a}. Pictures of {g} are forbidden; give the remaining tile.','Imagine keeping only the row of {a}, then deleting the {g} photos. Choose the photo left over.'],
'neighbor_row_filter':['Choose the {g} photo in the same row as the tile immediately {d} of {a}.','From {a}, move one tile {d}, then find {g} in the row you reach.','Use the immediate {n} neighbor of {a} as a reference; select the {g} photo in its row.','First find {a}, then step one tile {d}. Which {g} photo shares that row?','Select {g}, searching only within the row reached by one step {d} from {a}.','The search row contains the immediate {n} neighbor of {a}. Answer with the {g} picture in that row.'],
'neighbor_col_filter':['Choose the {g} photo in the same column as the tile immediately {d} of {a}.','From {a}, move one tile {d}, then find {g} in the column you reach.','Use the immediate {n} neighbor of {a} as a reference; select the {g} photo in its column.','First find {a}, then step one tile {d}. Which {g} photo shares that column?','Select {g}, searching only within the column reached by one step {d} from {a}.','The search column contains the immediate {n} neighbor of {a}. Answer with the {g} picture in that column.'],
'row_filter_then_neighbor':['Find the {g} photo in the same row as {a}, then choose the photo one tile {d} of it.','Find {g} in the horizontal row containing {a}; select its immediate {n} neighbor.','First locate {g} in the row of {a}. Choose the tile immediately {d} of that photo.','Which photo is one tile {d} of the {g} picture sharing a row with {a}?','Your starting tile shows {g} and shares a row with {a}. Move one tile {d} from that starting tile.','The final photo is the immediate {n} neighbor of a {g} photo. Find that {g} photo in the row of {a}.'],
}
def instruction(program,language,variant,seed=0):
    rng=random.Random(seed)
    def word(key):return rng.choice((CATALOG if key in CATALOG else GROUPS)[key][language])
    data={alias:word(program[key]) for key,alias in [('target','t'),('anchor','a'),('group','g'),('other_group','h'),('yes','y'),('no','z')] if key in program}
    if 'threshold' in program:data['k']=str(program['threshold'])
    if 'direction' in program:
        d=program['direction'];op={'left':'right','right':'left','up':'down','down':'up'}[d]
        if language=='ko':
            movement={'left':'왼쪽','right':'오른쪽','up':'위쪽','down':'아래쪽'}
            data.update(d=movement[d],n=movement[d],opn=movement[op])
        else:
            movement={'left':'to the left','right':'to the right','up':'up','down':'down'}
            noun={'left':'left','right':'right','up':'upper','down':'lower'}
            data.update(d=movement[d],n=noun[d],opn=noun[op])
    if 'second_direction' in program:
        data['e']=({'left':'왼쪽','right':'오른쪽','up':'위쪽','down':'아래쪽'} if language=='ko' else {'left':'to the left','right':'to the right','up':'up','down':'down'})[program['second_direction']]
    template=(KO if language=='ko' else EN)[program['family']][variant]
    if language=='en' and 'direction' in program:
        data['rel']={'left':'to the left of','right':'to the right of','up':'above','down':'below'}[program['direction']]
        template=template.replace('{d} of','{rel}').replace('a non-{g} photo','a photo outside the {g} category')
    if language=='ko':
        for key,value in data.items():
            last=value[-1]
            final=('가'<=last<='힣') and (ord(last)-ord('가'))%28!=0
            # Copula is 이야/야, not the subject-particle pair 이/가.
            if not final:template=template.replace('{'+key+'}이야','{'+key+'}야')
            for consonant,vowel in [('이','가'),('을','를'),('과','와')]:
                template=template.replace('{'+key+'}'+consonant,'{'+key+'}'+(consonant if final else vowel))
    text=template.format(**data)
    # Correct Korean object particles for these authored fragments; no semantic
    # substitutions, generated answers, or model-generated paraphrases.
    if language=='ko':
        prefixes=['','이 사진판을 보고 답해 줘. ','부탁 하나 할게. ','다음 조건에 맞춰 사진을 골라 줘. ']
        suffixes=['',' 해당하는 사진이 없으면 NONE이야.',' 칸을 벗어나는 이동은 NONE으로 답해.']
    else:
        prefixes=['','Look at this photo board. ','Please help me select one tile. ','Follow this request: ']
        suffixes=['',' Choose NONE if no tile qualifies.',' A move outside the grid means NONE.']
    return rng.choice(prefixes)+text+rng.choice(suffixes)
