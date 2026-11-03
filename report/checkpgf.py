import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    reports = []
    for path in sorted((ROOT/'figures').glob('fig*.pdf')):
        xml = subprocess.check_output(['pdftotext', '-bbox', str(path), '-'], text=True)
        xml = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', xml)
        tree = ET.fromstring(xml)
        pages = list(tree.iter('{http://www.w3.org/1999/xhtml}page'))
        assert len(pages) == 1, path
        page = pages[0]; width, height = float(page.attrib['width']), float(page.attrib['height'])
        assert abs(width-468) < 1, (path, width)
        clipped = []
        for word in page:
            a = word.attrib
            if float(a['xMin']) < -.5 or float(a['yMin']) < -.5 or float(a['xMax']) > width+.5 or float(a['yMax']) > height+.5:
                clipped.append(word.text)
        assert not clipped, (path, clipped)
        fonts = subprocess.check_output(['pdffonts', str(path)], text=True)
        assert 'Type 3' not in fonts, path
        reports.append(dict(figure=path.name, width_pdf_pt=width, height_pdf_pt=height,
                            clipped_words=clipped, rasterized_fonts=False))
    assert len(reports) == 10
    target = ROOT/'figures/pgf_readability_checks.json'
    target.write_text(json.dumps(reports, indent=2)+'\n')
    print('Ten vector figures: page bounds and font checks passed.')

if __name__ == '__main__':
    main()
