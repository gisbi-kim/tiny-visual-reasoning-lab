"""Render the final Markdown article into a small, self-contained Pages site."""
import html, json, re, shutil, struct
from pathlib import Path
from urllib.parse import quote, urlparse
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent

def main():
    source = ROOT/'docs/BLOG.md'
    text = source.read_text(encoding='utf-8')
    for marker in ('평가 후 입력', '작성 중인 원고', '최종 글에는', '최종 원고에서는'):
        if marker in text:
            raise RuntimeError(f'Finish the article before publishing: {marker}')
    site = ROOT/'site'
    site.mkdir(exist_ok=True)
    (site/'assets').mkdir(exist_ok=True)
    md = MarkdownIt('commonmark', {'html': True}).enable('table')
    tokens = md.parse(text)
    toc = []
    for index, token in enumerate(tokens):
        if token.type == 'heading_open' and token.tag == 'h2':
            anchor = f'section-{len(toc)+1}'
            token.attrSet('id', anchor)
            toc.append((anchor, tokens[index+1].content))
        elif token.type == 'heading_open' and token.tag == 'h3' and tokens[index+1].content.startswith('12.4 이 예제를 벗어나서:'):
            token.attrSet('id', 'decision-api-reference')
        elif token.type == 'heading_open' and token.tag == 'h3' and tokens[index+1].content.startswith('12.5 Decision 모델을 학습할 때는'):
            token.attrSet('id', 'decision-training-reference')
        for child in token.children or []:
            if child.type == 'image':
                src = child.attrGet('src') or ''
                if urlparse(src).scheme:
                    raise RuntimeError('Use local verified figures in the article')
                path = (source.parent/src).resolve()
                if not path.is_relative_to(ROOT) or not path.is_file():
                    raise RuntimeError(f'Figure not found: {src}')
                shutil.copy2(path, site/'assets'/path.name)
                child.attrSet('src', f'assets/{path.name}')
                child.attrSet('loading', 'lazy')
                if path.suffix.lower() == '.png':
                    width, height = struct.unpack('>II', path.read_bytes()[16:24])
                    child.attrSet('width', str(width))
                    child.attrSet('height', str(height))
                    child.attrSet('loading', 'eager')
            elif child.type == 'link_open':
                href = child.attrGet('href') or ''
                if href and not urlparse(href).scheme and not href.startswith('#'):
                    path = (source.parent/href).resolve()
                    if not path.is_relative_to(ROOT) or not path.exists():
                        raise RuntimeError(f'Linked artifact not found: {href}')
                    rel = quote(path.relative_to(ROOT).as_posix(), safe='/')
                    kind = 'tree' if path.is_dir() else 'blob'
                    child.attrSet('href', f'https://github.com/gisbi-kim/tiny-visual-reasoning-lab/{kind}/main/{rel}')
    body = md.renderer.render(tokens, md.options, {})
    navigation = ''.join(f'<li><a href="#{a}">{html.escape(t)}</a></li>' for a, t in toc)
    page = '''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Tiny ImageNet과 Qwen3.5-4B 기반 decision model의 데이터 설계, LoRA 학습, v1·v2 비교와 실제 실험 기록.">
<title>사진 여섯 장과 자연어 조건으로 decision model 학습하기</title>
<style>
:root{color-scheme:light;--ink:#24313d;--muted:#66727c;--accent:#226b88}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#fafbf9;color:var(--ink);font:17px/1.85 system-ui,-apple-system,"Malgun Gothic",sans-serif;word-break:keep-all;overflow-wrap:anywhere}header{border-bottom:1px solid #dce1df;padding:20px 5vw;font-size:13px;color:var(--muted)}header a{font-weight:600}main{display:grid;grid-template-columns:230px minmax(0,820px);gap:48px;max-width:1200px;margin:50px auto 90px;padding:0 28px}nav{position:sticky;top:24px;align-self:start;max-height:90vh;overflow:auto;font-size:13px}nav ol{list-style:none;padding:0}nav li{margin:9px 0}nav a{color:var(--muted);text-decoration:none}h1{font-size:34px;line-height:1.45;letter-spacing:-.045em;margin:0 0 28px}h2{font-size:25px;line-height:1.5;letter-spacing:-.025em;margin:64px 0 20px;scroll-margin-top:30px}h3{font-size:20px;margin-top:35px}p{margin:18px 0}a{color:var(--accent);text-underline-offset:3px}strong{font-weight:700}table{display:block;overflow-x:auto;border-collapse:collapse;width:100%;margin:24px 0;font-size:14px;white-space:normal}th,td{padding:10px 14px;border-bottom:1px solid #dce1df;text-align:left;min-width:90px}th{background:#edf2f2;font-weight:650}td ul{margin:0;padding-left:1.3em}td li{padding-left:.2em;margin:.3em 0}pre{padding:20px;background:#edf1f2;border:1px solid #dce1df;border-radius:6px;overflow-x:auto;font-size:13px;line-height:1.7}code{font-family:ui-monospace,Consolas,monospace}p code,li code{background:#edf1f2;padding:2px 4px;border-radius:3px;font-size:.9em}img{max-width:100%;height:auto;display:block;margin:28px auto;background:white}img[src$=".png"]{width:576px;border-radius:4px}img[src*="user-test-"]{width:100%}blockquote{margin:25px 0;border-left:3px solid #9cb6bf;padding:2px 22px;color:#55636d}article>ul,article>ol{padding-left:25px}footer{border-top:1px solid #dce1df;margin-top:70px;padding-top:20px;font-size:13px;color:var(--muted)}@media(max-width:950px){main{display:block;max-width:830px}nav{position:static;max-height:none;border-bottom:1px solid #dce1df;margin-bottom:35px;padding-bottom:16px}nav ol{columns:2}}@media(max-width:560px){body{font-size:16px}main{padding:0 18px;margin-top:28px}h1{font-size:28px}h2{font-size:23px;margin-top:48px}nav ol{columns:1}th,td{padding:9px 10px}pre{padding:14px}}
</style></head><body><header>Tiny Visual Reasoning Lab · 기술 실험 기록 · <a href="https://github.com/gisbi-kim/tiny-visual-reasoning-lab">코드와 실험 자료</a></header>
<main><nav aria-label="목차"><strong>목차</strong><ol>''' + navigation + '''</ol></nav><article>''' + body + '''<footer>그림과 표는 저장된 실제 실험 결과에서 생성했습니다. 원본 설정·예측·로그는 본문의 GitHub 링크에서 확인할 수 있습니다.</footer></article></main></body></html>'''
    (site/'index.html').write_text(page, encoding='utf-8')
    (site/'.nojekyll').write_text('', encoding='utf-8')
    print('BLOG_SITE_READY', site/'index.html', flush=True)

if __name__ == '__main__':
    main()
