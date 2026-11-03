"""Create standalone SVG figures from completed records using the Pygal library."""
import importlib.metadata
import json, sys
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
DOC_DEPS = ROOT/'.cache/chart-deps'
if DOC_DEPS.exists():
    sys.path.insert(0, str(DOC_DEPS))
import pygal
from pygal.style import Style

SVG = 'http://www.w3.org/2000/svg'
ET.register_namespace('', SVG)
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
STYLE = Style(background='white', plot_background='white', foreground='#24313d',
              foreground_strong='#24313d', foreground_subtle='#dce1df',
              colors=('#8395a7', '#226b88', '#b65c35'), font_family='Arial, sans-serif',
              title_font_size=20, label_font_size=13, major_label_font_size=13,
              legend_font_size=14, value_font_size=16, opacity=1, stroke_width=2)

def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))

def svg_element(chart):
    element = ET.fromstring(chart.render())
    # Static figures do not need the optional Pygal tooltip configuration.
    for parent in element.iter():
        for child in list(parent):
            if child.tag == f'{{{SVG}}}script':
                parent.remove(child)
    return element

def save(element, name):
    output = ROOT/'docs/figures'
    output.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(element).write(output/f'{name}.svg', encoding='utf-8', xml_declaration=True)

def main():
    config = dict(style=STYLE, js=[], width=960, height=500, explicit_size=True,
                  show_x_guides=False, show_y_guides=True, truncate_label=-1,
                  legend_at_bottom=True, legend_at_bottom_columns=2)
    records = read('results/v2/original-suite-comparison.json')['suites']
    keys = ['test_language', 'test_composition', 'test_combined']
    chart = pygal.Bar(**config, title='Same original test examples: v1 vs v2', range=(0, 100),
                      y_title='Accuracy (%)', print_values=True, print_values_position='top',
                      value_formatter=lambda value: f'{value:.2f}%')
    chart.x_labels = ['Wording held out in v1', 'Compositions held out in v1', 'Both held out in v1']
    chart.add('v1', [100*records[k]['v1_accuracy'] for k in keys])
    chart.add('v2', [100*records[k]['v2_accuracy'] for k in keys])
    save(svg_element(chart), 'original-suite-comparison')
    history = [r for r in read('docs/v2/training-history.json') if 'loss' in r and 'step' in r]
    validation = read('docs/v2/validation-history.json')
    best = max(validation, key=lambda r: r['accuracy'])
    config['height'] = 360
    loss = pygal.XY(**config, title='V2 continuation training loss', x_title='Additional training steps',
                    y_title='Loss', show_dots=False)
    loss.add('Recorded loss', [(r['step'], r['loss']) for r in history])
    window = min(20, len(history))
    loss.add(f'Mean over {window} logged points', [(history[i]['step'],
        sum(r['loss'] for r in history[i-window+1:i+1])/window) for i in range(window-1, len(history))])
    valid = pygal.XY(**config, title=f"Validation accuracy (selected checkpoint: step {best['step']})",
                     x_title='Additional training steps', y_title='Accuracy (%)', range=(80, 100),
                     value_formatter=lambda value: f'{value:.2f}')
    valid.add('336 validation examples', [(r['step'], 100*r['accuracy']) for r in validation])
    valid.add('Selected checkpoint', [(best['step'], 100*best['accuracy'])], dots_size=6)
    combined = ET.Element(f'{{{SVG}}}svg', {'width': '960', 'height': '720', 'viewBox': '0 0 960 720',
                                         'role': 'img', 'aria-label': 'V2 training loss and validation accuracy'})
    for i, panel in enumerate((loss, valid)):
        element = svg_element(panel)
        element.attrib.update(x='0', y=str(i*360), width='960', height='360')
        combined.append(element)
    save(combined, 'v2-training-history')
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions(path=[str(DOC_DEPS)])}
    (ROOT/'docs/v2/docs-environment.json').write_text(json.dumps({'python': sys.version, 'packages': packages,
        'install_scope': 'Separate project-local .cache/chart-deps; shared training environment not modified.',
        'plotting_note': 'Pygal generates standalone SVG. Matplotlib DLL loading was blocked by Windows application-control policy; that policy was not changed.'}, indent=2), encoding='utf-8')
    print('PLOTS_READY', flush=True)

if __name__ == '__main__':
    main()
