from prepare_v2 import surface,FAMILIES,FORMS

def test_surface_variations_preserve_numerals_and_korean_particles():
    original='위쪽으로 이동해. 동물 사진이 3장 이상이면 골라.'
    forms={surface(original,'ko',s) for s in range(30)}
    assert any('세 장' in s for s in forms)
    assert any('3장' in s for s in forms)
    assert any('위로 이동' in s for s in forms)
    assert not any('위으로' in s for s in forms)
    english={surface('If at least 3 pictures show animals, choose the cat.','en',s) for s in range(30)}
    assert any('three pictures' in s for s in english)
    assert any('3 pictures' in s for s in english)
    assert surface(original,'ko',123)==surface(original,'ko',123)

def test_no_withheld_families_or_forms_in_v2():
    assert len(FAMILIES)==12 and len(set(FAMILIES))==12
    assert FORMS==(0,1,2,3,4,5)
