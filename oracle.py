"""Exact program execution for data construction/evaluation; never inference."""
from catalog import matches
from schema import ACTIONS
TRAIN_FAMILIES=('direct','neighbor','row_filter','col_filter','neighbor_filter','chain','count_branch','compare_count','row_except','neighbor_row_filter')
HELDOUT_FAMILIES=('neighbor_col_filter','row_filter_then_neighbor')
DIRECTIONS=('left','right','up','down')
def unique(indices):
    return indices[0] if len(indices)==1 else None if not indices else 'AMBIGUOUS'
def adjacent(index,direction):
    if index is None:return None
    row,col=divmod(index,3)
    dr,dc={'left':(0,-1),'right':(0,1),'up':(-1,0),'down':(1,0)}[direction]
    row,col=row+dr,col+dc
    return row*3+col if 0<=row<2 and 0<=col<3 else None
def execute(board,program):
    f=program['family'];trace=[]
    def find(selector):return unique([i for i,k in enumerate(board) if matches(k,selector)])
    if f=='direct':result=find(program['target'])
    elif f in ('count_branch','compare_count'):
        count=sum(matches(k,program['group']) for k in board)
        rhs=program['threshold'] if f=='count_branch' else sum(matches(k,program['other_group']) for k in board)
        condition=count>=rhs if f=='count_branch' else count>rhs
        trace.append({'count':count,'comparison_value':rhs,'condition':condition})
        result=find(program['yes'] if condition else program['no'])
    else:
        anchor=find(program['anchor'])
        if anchor=='AMBIGUOUS':return None,trace
        trace.append({'anchor':anchor})
        if f=='neighbor':result=adjacent(anchor,program['direction'])
        elif f=='chain':
            intermediate=adjacent(anchor,program['direction'])
            trace.append({'intermediate':intermediate})
            result=adjacent(intermediate,program['second_direction'])
        elif f=='neighbor_filter':
            candidate=adjacent(anchor,program['direction'])
            result=candidate if candidate is not None and matches(board[candidate],program['group']) else None
        else:
            intermediate=adjacent(anchor,program['direction']) if f.startswith('neighbor_') else anchor
            trace.append({'intermediate':intermediate})
            is_row=f not in ('col_filter','neighbor_col_filter')
            indices=[] if intermediate is None else [i for i in range(6) if (i//3==intermediate//3 if is_row else i%3==intermediate%3)]
            candidates=[i for i in indices if matches(board[i],program['group']) != (f=='row_except')]
            trace.append({'candidates':candidates})
            result=unique(candidates)
            if f=='row_filter_then_neighbor' and result!='AMBIGUOUS':result=adjacent(result,program['direction'])
    if result=='AMBIGUOUS':return None,trace
    return ACTIONS[result] if result is not None else 'NONE',trace
