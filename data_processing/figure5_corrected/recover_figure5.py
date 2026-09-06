"""Recover vector chart coordinates and preserve original artwork with corrected titles.

Chart contract: two five-series line panels, five ESM2 sizes per series.
Output: original vector PDF artwork, corrected group mapping supplied by the user,
plus PNG previews, extracted numerical values and a QA receipt. Zero-shot is untouched.
Original palettes (including panel-specific legend mappings), fonts, scales and
markers are preserved deliberately because this is a faithful reproduction.
The PDF identifies the metric only as F1; pooled/macro and run provenance are unknown.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pdfplumber
import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, DecodedStreamObject, FloatObject, NameObject

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent
SIZES = ['8M', '35M', '150M', '650M', '3B']
MAPPING = [
    ('overrepresented', 'underrepresented_test_results.pdf', 'overrepresented_test_results.pdf'),
    ('underrepresented', 'overrepresented_test_results.pdf', 'underrepresented_test_results.pdf'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def title_index(ops):
    matches = [i for i, (a, o) in enumerate(ops) if o == b'TJ'
               and 'Test Set' in ''.join(v for v in a[0] if isinstance(v, str))]
    assert len(matches) == 1, matches
    return matches[0]


def extract(path):
    reader = PdfReader(path)
    page = reader.pages[0]
    ops = ContentStream(page.get_contents(), reader).operations
    stroke, stack, points, lines, grids, clips = None, [], [], [], [], []
    for args, op in ops:
        if op == b'q':
            stack.append(stroke)
        elif op == b'Q':
            stroke = stack.pop()
        elif op == b'RG':
            stroke = tuple(float(v) for v in args)
        elif op == b'G':
            stroke = (float(args[0]),) * 3
        elif op == b're':
            clips.append(tuple(float(v) for v in args))
            points = []
        elif op == b'm':
            points = [tuple(float(v) for v in args)]
        elif op == b'l':
            points.append(tuple(float(v) for v in args))
        elif op in (b'S', b's', b'f', b'f*', b'B', b'B*', b'n'):
            if op == b'S' and len(points) == 5 and stroke and max(stroke)-min(stroke) > .1:
                if all(points[i+1][0] > points[i][0] + 50 for i in range(4)):
                    lines.append({'rgb': stroke, 'points': points.copy()})
            if op == b'S' and len(points) == 2:
                if abs(points[0][1]-points[1][1]) < 1e-6 and points[1][0]-points[0][0] > 350:
                    grids.append(points[0][1])
            points = []
    assert len(lines) == 5, (path, len(lines))
    axes = next(rect for rect in clips if rect[2] > 400 and 300 < rect[3] < 350)
    with pdfplumber.open(path) as doc:
        words = doc.pages[0].extract_words()
        tick_words = [w for w in words if re.fullmatch(r'0\.\d+', w['text'])
                      and w['x1'] < axes[0]]
    anchors = []
    for word in tick_words:
        y = float(page.mediabox.height) - (word['top'] + word['bottom']) / 2
        grid = min(grids, key=lambda g: abs(g-y))
        assert abs(grid-y) < 10
        anchors.append((grid, float(word['text'])))
    assert len(anchors) >= 5
    a = np.array(anchors)
    slope, intercept = np.polyfit(a[:, 0], a[:, 1], 1)
    residual = float(np.max(np.abs(a[:, 1] - (slope*a[:, 0]+intercept))))
    assert residual < 1e-8, residual
    names = page.extract_text().split('Architecture\n', 1)[1].strip().splitlines()
    assert len(names) == 5, names
    for line, name in zip(lines, names):
        line['architecture'] = name
        line['values_geometry'] = [float(slope*y+intercept) for x, y in line['points']]
        line['values_4dp'] = [round(v, 4) for v in line['values_geometry']]
        assert max(abs(x-y) for x, y in zip(line['values_geometry'], line['values_4dp'])) < 1e-8
    return reader, {'file': str(path), 'sha256': sha(path), 'axes_rect': axes,
                    'tick_anchors_pdf_y_to_value': anchors, 'slope': float(slope),
                    'intercept': float(intercept), 'max_tick_fit_residual': residual,
                    'series': lines}


def render(path, destination, scale=2):
    doc = pdfium.PdfDocument(str(path))
    bitmap = doc[0].render(scale=scale)
    im = bitmap.to_pil().convert('RGB')
    im.save(destination)
    bitmap.close()
    doc.close()
    return im


def make_panel(group, source_name, donor_name, readers, info):
    writer = PdfWriter()
    writer.add_page(readers[source_name].pages[0])
    target = writer.pages[0]
    original_ops = ContentStream(target.get_contents(), writer).operations
    old_title = title_index(original_ops)
    edits = []
    if donor_name:
        donor = readers[donor_name].pages[0]
        donor_ops = ContentStream(donor.get_contents(), readers[donor_name]).operations
        donor_title = title_index(donor_ops)
        assert original_ops[old_title-4][1] == b'cm'
        assert original_ops[old_title-2][1] == b'Tf'
        font = donor['/Resources']['/Font']['/F1'].get_object().clone(writer)
        target['/Resources']['/Font'][NameObject('/CorrectedTitleFont')] = font
        # Reuse the opposite title's exact glyphs and kerning. Adjust its translation
        # by the difference in plot centers to retain centered title alignment.
        center = info[source_name]['axes_rect'][0] + info[source_name]['axes_rect'][2]/2
        donor_center = info[donor_name]['axes_rect'][0] + info[donor_name]['axes_rect'][2]/2
        translated = copy.deepcopy(donor_ops[donor_title-4][0])
        translated[4] = FloatObject(float(translated[4]) + center-donor_center)
        replacement = {
            old_title-4: (translated, b'cm'),
            old_title-2: ([NameObject('/CorrectedTitleFont'), FloatObject(20)], b'Tf'),
            old_title: (copy.deepcopy(donor_ops[donor_title][0]), b'TJ'),
        }
        # Patch only the title bytes. Reserializing every PDF operation rounds
        # some original color/clip floats in pypdf, so retain all other bytes.
        pattern = re.compile(rb'q\s+1 0 -?0 1 ([\d.]+) ([\d.]+) cm\s+BT\s+/F1 20 Tf\s+0 0 Td\s+(\[.*?\])\s+TJ\s+ET\s+Q', re.S)
        original_data = target.get_contents().get_data()
        donor_data = donor.get_contents().get_data()
        old_matches = list(pattern.finditer(original_data))
        donor_matches = list(pattern.finditer(donor_data))
        assert len(old_matches) == len(donor_matches) == 1
        dm = donor_matches[0]
        x = float(dm.group(1)) + center-donor_center
        new_title = (f'q\n1 0 0 1 {x:.10f} {float(dm.group(2)):.10f} cm\nBT\n/CorrectedTitleFont 20 Tf\n0 0 Td\n'.encode()
                     + dm.group(3) + b' TJ\nET\nQ')
        changed_data = original_data[:old_matches[0].start()] + new_title + original_data[old_matches[0].end():]
        stream = DecodedStreamObject()
        stream.set_data(changed_data)
        target.replace_contents(stream)
        cs = ContentStream(target.get_contents(), writer)
        edits = list(replacement)
        # The only modified operations are the title's position, font and glyphs.
        assert all(a == b for i, (a, b) in enumerate(zip(original_ops, cs.operations)) if i not in edits)
    writer.add_metadata({'/Title': group.replace('_', '-') + ' test results - corrected panel label',
                         '/Subject': 'Vector artwork preserved. Group labels reassigned at user request; F1 aggregation unverified.'})
    out = ROOT / f'{group}_test_results_corrected.pdf'
    with out.open('wb') as stream:
        writer.write(stream)
    new_text = PdfReader(out).pages[0].extract_text().replace('T est', 'Test')
    expected = {'overrepresented': 'Overrepresented Test Set',
                'underrepresented': 'Underrepresented Test Set', 'zero_shot': 'Zero-shot Test Set'}[group]
    assert expected in new_text, new_text
    _, reextracted = extract(out)
    assert reextracted['series'] == info[source_name]['series']
    assert reextracted['axes_rect'] == info[source_name]['axes_rect']
    old_img = render(SOURCE/source_name, ROOT/'qa'/f'{group}_source.png')
    new_img = render(out, ROOT/f'{group}_test_results_corrected.png')
    old_a, new_a = np.asarray(old_img), np.asarray(new_img)
    # Exclude only title band above y=400 PDF points (page is 432 points high).
    start = 64
    unchanged = bool(np.array_equal(old_a[start:], new_a[start:]))
    assert unchanged, f'Non-title pixels changed for {group}'
    return out, {'modified_content_operations': edits,
                 'series_and_axes_exactly_preserved': True,
                 'render_pixels_below_title_exactly_preserved': unchanged}


def main():
    (ROOT/'qa').mkdir(exist_ok=True)
    readers, info = {}, {}
    for _, filename, _ in MAPPING:
        readers[filename], info[filename] = extract(SOURCE/filename)
    (ROOT/'vector_extraction.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    fields = ['corrected_group', 'source_file', 'source_group_label', 'architecture',
              'esm2_size', 'plotted_f1_4dp', 'geometry_recovered_f1', 'pdf_x', 'pdf_y',
              'rgb', 'metric_aggregation', 'group_mapping_basis']
    with (ROOT/'extracted_plot_values.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for group, filename, _ in MAPPING:
            for series in info[filename]['series']:
                for size, val, raw, point in zip(SIZES, series['values_4dp'], series['values_geometry'], series['points']):
                    w.writerow(dict(corrected_group=group, source_file=filename,
                        source_group_label=filename.replace('_test_results.pdf', ''),
                        architecture=series['architecture'], esm2_size=size,
                        plotted_f1_4dp=f'{val:.4f}', geometry_recovered_f1=f'{raw:.12f}',
                        pdf_x=point[0], pdf_y=point[1], rgb=str(series['rgb']),
                        metric_aggregation='not established by PDF; source axis is F1',
                        group_mapping_basis='user-requested reversal of represented-group labels'))
    panels, checks = [], {}
    for group, source, donor in MAPPING:
        panel, check = make_panel(group, source, donor, readers, info)
        panels.append(panel)
        checks[group] = check
    for name, source in info.items():
        assert sha(Path(source['file'])) == source['sha256'], name
    qa = {'data_points': 50, 'series': 10, 'panels': checks,
          'source_hashes_unchanged': True,
          'maximum_4dp_recovery_error': max(abs(v-round(v,4)) for src in info.values() for line in src['series'] for v in line['values_geometry']),
          'outputs': {p.name: sha(p) for p in [*panels, ROOT/'extracted_plot_values.csv']}}
    (ROOT/'verification.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print(json.dumps(qa, indent=2))
    print('RECOVERED 650M VALUES')
    for group, filename, _ in MAPPING:
        print(group, {line['architecture']: line['values_4dp'][3] for line in info[filename]['series']})


if __name__ == '__main__':
    main()
