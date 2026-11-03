"""Model inputs deliberately exclude oracle, object labels and task programs."""
ACTIONS=tuple('ABCDEF')+('NONE',)
def questions():
    positions=('top left','top middle','top right','bottom left','bottom middle','bottom right')
    return {'select':{'type':'choice','instructions':
        'Follow the user instruction using the six photographs arranged in a 2-row, 3-column grid. '
        'Each photograph is represented by its main object. Adjacent means immediately next to a tile, without wrapping. '
        'Choose a tile, or NONE if no tile qualifies.',
        'criteria':{**{a:f'Choose the {p} photograph.' for a,p in zip(ACTIONS,positions)},'NONE':'No photograph satisfies the instruction.'}}}
def model_input(image,instruction):
    if not isinstance(instruction,str) or not instruction.strip():raise ValueError('Enter an instruction.')
    return {'state':instruction,'images':[image],'questions':questions()}
