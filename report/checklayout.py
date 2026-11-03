import importlib
import json
from pathlib import Path
from matplotlib.text import Text
from matplotlib.backends.backend_agg import FigureCanvasAgg
import mplfig as figures


def inspect(fig, name):
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    texts, clipped, overlaps = [], [], []
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text().strip():
            box = text.get_window_extent(renderer)
            if box.width > 0 and box.height > 0:
                texts.append((text.get_text(), box, text.get_fontsize()))
                if (box.x0 < -1 or box.y0 < -1 or
                        box.x1 > width + 1 or box.y1 > height + 1):
                    clipped.append(text.get_text())
    for index, (first, a, _) in enumerate(texts):
        for second, b, _ in texts[index + 1:]:
            if (min(a.x1, b.x1) - max(a.x0, b.x0) > 2 and
                    min(a.y1, b.y1) - max(a.y0, b.y0) > 2):
                overlaps.append([first, second])
    return dict(figure=name, minimum_font_pt=min(t[2] for t in texts),
                clipped_text=clipped, overlapping_text=overlaps)


def main():
    root = Path(__file__).resolve().parent.parent
    results = json.loads((root / 'results/summary.json').read_text())
    reports = []
    save = figures.save

    def checked_save(fig, name):
        reports.append(inspect(fig, name))
        save(fig, name)

    figures.save = checked_save
    figures.fig1_reliability(results)
    figures.fig2_efficiency(results)
    figures.fig4_noise_sweep(results)
    figures.fig3_limits(results)
    audit = importlib.import_module('solverfig')
    reports.append(inspect(audit.fig, 'fig_revision_audit'))
    target = root / 'figures/readability_checks.json'
    target.write_text(json.dumps(reports, indent=2) + '\n')
    for report in reports:
        assert report['minimum_font_pt'] >= 9.5, report
        assert not report['clipped_text'], report
        assert not report['overlapping_text'], report
    print('All five figures pass font, clipping and text-collision checks.')
    print(target)


if __name__ == '__main__':
    main()
