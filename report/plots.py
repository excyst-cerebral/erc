import json
import subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'figures'
TEX = OUT / 'pgf'
TEX.mkdir(parents=True, exist_ok=True)
RESULTS = json.loads((ROOT / 'results/summary.json').read_text())
REVISION = json.loads((ROOT / 'results/validate.json').read_text())
REVIEW = json.loads((ROOT / 'results/audit.json').read_text())
WIDTH = 469.755
MANIFEST = {}
PALETTE = {'navy': '1F5A85', 'teal': '347F78', 'rust': 'B96332',
           'purple': '80678F', 'slate': '737E87', 'ink': '30363B',
           'grid': 'E6E9EC', 'reference': '737B82', 'paperblue': 'EEF3F7'}
METHODS = ['naive', 'known_T', 'uniform_shift', 'sharp_suspect',
           'sharp_oracle_flags', 'oracle_clean']
NAMES = ['Naive CP', 'Known-$T$', 'Rank shift', 'Exact (flags)',
         'Exact (true flags)', 'Clean-label CP']
STYLES = {
    'naive': 'color=rust,dashed,mark=o,mark options={solid,fill=white},line width=.85pt',
    'known_T': 'color=purple,densely dashed,mark=square,mark options={solid,fill=white},line width=.85pt',
    'uniform_shift': 'color=slate,dashdotted,mark=triangle*,mark options={solid},line width=.85pt',
    'sharp_suspect': 'color=navy,solid,mark=*,mark options={solid},line width=1.15pt',
    'sharp_oracle_flags': 'color=teal,densely dashdotted,mark=diamond,mark options={solid,fill=white},line width=.9pt',
    'oracle_clean': 'color=ink,densely dotted,mark=x,mark options={solid},line width=.9pt',
}
PREAMBLE = r'''\documentclass[tikz,border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,amssymb}
\usepackage{pgfplots}
\usetikzlibrary{arrows.meta,positioning,calc}
\pgfplotsset{compat=1.18}
\newcommand{\figfont}{\sffamily\fontsize{9.5}{11}\selectfont}
\newcommand{\panelfont}{\sffamily\fontsize{10}{12}\selectfont}
\pgfplotsset{paper axis/.style={scale only axis,axis lines*=left,
 axis line style={draw=ink,line width=.45pt},
 tick align=outside,tick style={draw=ink,line width=.45pt},
 ticklabel style={font=\figfont,text=ink},label style={font=\figfont,text=ink},
 title style={font=\panelfont,text=ink},
 ymajorgrids=true,grid style={draw=grid,line width=.3pt},
 scaled ticks=false,clip=true,unbounded coords=jump,
 every axis plot/.append style={mark size=1.7pt},
 /pgf/number format/1000 sep={\,}}}
'''
PREAMBLE += '\n'.join(fr'\definecolor{{{name}}}{{HTML}}{{{value}}}' for name, value in PALETTE.items())


def number(x):
    return f'{float(x):.10g}'


def wrap(body, height):
    return (PREAMBLE + '\n\\begin{document}\n\\begin{tikzpicture}[font=\\figfont]\n'
            + fr'\path[use as bounding box] (0,0) rectangle ({WIDTH}pt,{height}pt);'
            + '\n' + body + '\n\\end{tikzpicture}\n\\end{document}\n')


def axis(x, y, width, height, options, body):
    return (fr'\begin{{axis}}[paper axis,at={{({x}pt,{y}pt)}},anchor=south west,'
            + fr'width={width}pt,height={height}pt,{options}]' + '\n'
            + body + '\n\\end{axis}\n')


def panel(x, y, text):
    return fr'\node[anchor=west,font=\panelfont,text=ink] at ({x}pt,{y}pt) {{{text}}};' + '\n'


def line(method, points, horizontal=False):
    if not points:
        return ''
    coords = ' '.join((fr'({number(x)},{number(y)}) +- ({number(sd)},0)' if horizontal
                       else fr'({number(x)},{number(y)}) +- (0,{number(sd)})') for x, y, sd in points)
    direction = 'x' if horizontal else 'y'
    options = STYLES[method] + (',only marks' if horizontal else '')
    return (fr'\addplot+[{options},error bars/.cd,{direction} dir=both,{direction} explicit,'
            + r'error bar style={solid,line width=.6pt},error mark options={rotate=90,mark size=2pt}] coordinates {'
            + coords + '};\n')


def legend(y=22, methods=None, names=None):
    methods = METHODS if methods is None else methods
    names = NAMES if names is None else names
    result = ''
    for i, (method, name) in enumerate(zip(methods, names)):
        x = [42, 183, 330][i % 3]
        yy = y - (i // 3) * 15
        result += (fr'\draw[{STYLES[method]}] ({x}pt,{yy}pt)--({x+18}pt,{yy}pt);'
                   + '\n' + fr'\draw[{STYLES[method]},mark size=1.7pt] plot coordinates {{({x+9}pt,{yy}pt)}};'
                   + '\n' + fr'\node[anchor=west,text=ink] at ({x+23}pt,{yy}pt) {{{name}}};' + '\n')
    return result


def aggregate(rows, field, xfield, xs):
    output = []
    for x in xs:
        values = [r[field] for r in rows if r[xfield] == x and field in r]
        if values:
            output.append((x, float(np.mean(values)), float(np.std(values))))
    return output


def reliability():
    name = 'fig1_reliability'
    config = [('exp1_controlled', lambda r: r['dataset']=='mnist' and r['eta_bar']==.2 and r['n_cal']==2000, '(a) MNIST'),
              ('exp1_controlled', lambda r: r['dataset']=='fmnist' and r['eta_bar']==.2 and r['n_cal']==2000, '(b) Fashion-MNIST'),
              ('exp2_cifar10n', lambda r: r['noisy']=='aggre_label' and r['n_cal']==2000 and r['budget_frac']==.2, '(c) CIFAR-10N')]
    body = ''
    MANIFEST[name] = []
    for i, (key, select, title) in enumerate(config):
        rows = [r for r in RESULTS[key] if select(r)]
        curves = ''
        for method in METHODS:
            points = []
            for alpha in [.2, .1, .05]:
                vals = [r[method+'_coverage'] for r in rows if r['alpha']==alpha and method+'_coverage' in r]
                if vals:
                    points.append((1-alpha, float(np.mean(vals)), float(np.std(vals))))
            curves += line(method, points)
            MANIFEST[name].append(dict(panel=i, method=method, points=points))
        curves = r'\addplot[reference,dashed,line width=.6pt] coordinates {(.775,.775) (.975,.975)};' + '\n' + curves
        opts = r'xmin=.775,xmax=.975,ymin=.60,ymax=1.025,xtick={.8,.9,.95},ytick={.6,.7,.8,.9,1},xlabel={Nominal $1-\alpha$}'
        opts += ',ylabel={Coverage}' if i == 0 else ',yticklabels={}'
        body += axis(42+i*145, 66, 128, 130, opts, curves)
        body += panel(42+i*145, 211, title)
    return name, wrap(body + legend(), 225)


def efficiency():
    name = 'fig2_efficiency'
    config = [('exp1_controlled', lambda r: r['dataset']=='mnist' and r['alpha']==.1 and r['n_cal']==2000, 'eta_bar', [.1,.2,.4], '(a) MNIST'),
              ('exp1_controlled', lambda r: r['dataset']=='fmnist' and r['alpha']==.1 and r['n_cal']==2000, 'eta_bar', [.1,.2,.4], '(b) Fashion-MNIST'),
              ('exp2_cifar10n', lambda r: r['alpha']==.1 and r['n_cal']==2000 and r['budget_frac']==.2, 'noisy', ['aggre_label','random_label1','worse_label'], '(c) CIFAR-10N')]
    body = ''
    MANIFEST[name] = []
    for i, (key, select, field, xs, title) in enumerate(config):
        rows = [r for r in RESULTS[key] if select(r)]
        curves = r'\addplot[reference,dashed,line width=.6pt] coordinates {(-.3,10) (2.3,10)};' + '\n'
        for method in METHODS:
            points = []
            for j, x in enumerate(xs):
                vals = [r[method+'_size'] for r in rows if r[field]==x and method+'_size' in r]
                if vals:
                    points.append((j, float(np.mean(vals)), float(np.std(vals))))
            curves += line(method, points)
            MANIFEST[name].append(dict(panel=i, method=method, points=points))
        opts = 'xmin=-.35,xmax=2.35,ymin=0,ymax=10.7,ytick={0,5,10},xtick={0,1,2}'
        opts += (r',xticklabels={10\%,20\%,40\%},xlabel={Noise $\bar\eta$}' if i < 2 else
                 r',xticklabels={{Agg.\\9\%},{Ann.~1\\17\%},{Worse\\40\%}},xticklabel style={align=center},xlabel={Human-label split}')
        opts += ',ylabel={Mean set size}' if i == 0 else ',yticklabels={}'
        body += axis(42+i*145, 73, 128, 123, opts, curves)
        body += panel(42+i*145, 211, title)
    return name, wrap(body + legend(), 225)


def noise_sweep():
    name = 'fig4_noise_sweep'
    rows = [r for r in RESULTS['exp1_controlled'] if r['dataset']=='mnist' and r['alpha']==.1 and r['n_cal']==2000]
    body = ''
    MANIFEST[name] = []
    for i, suffix in enumerate(['coverage', 'size']):
        target = .9 if i == 0 else 10
        curves = fr'\addplot[reference,dashed,line width=.6pt] coordinates {{(.065,{target}) (.435,{target})}};' + '\n'
        for method in METHODS:
            points = aggregate(rows, method+'_'+suffix, 'eta_bar', [.1,.2,.4])
            curves += line(method, points)
            MANIFEST[name].append(dict(panel=i, method=method, points=points))
        opts = r'xmin=.065,xmax=.435,xtick={.1,.2,.4},xlabel={Noise parameter $\bar\eta$},'
        opts += ('ymin=.68,ymax=1.025,ytick={.7,.8,.9,1},ylabel={Coverage}' if i==0 else
                 'ymin=0,ymax=10.8,ytick={0,5,10},ylabel={Mean set size}')
        body += axis([42,282][i], 66, 178, 130, opts, curves)
        body += panel([42,282][i], 211, ['(a) Coverage','(b) Set size'][i])
    return name, wrap(body+legend(), 225)


def diagnostics():
    name = 'fig3_limits'
    body = panel(129, 321, '(a) Model-agreement attack')
    rows = [r for r in RESULTS['exp3_adversarial'] if r['alpha']==.1 and r['eta_bar']==.1]
    lo, hi = rows[0]['rank_bound_low_fixed'], rows[0]['rank_bound_high_fixed']
    assert all(r['rank_bound_low_fixed']==lo and r['rank_bound_high_fixed']==hi for r in rows)
    curves = (fr'\path[fill=paperblue] (axis cs:{lo},-.6) rectangle (axis cs:{hi},6);' + '\n'
              + r'\addplot[reference,dashed,line width=.6pt] coordinates {(.9,-.6) (.9,6)};' + '\n'
              + fr'\node[text=reference,anchor=center] at (axis cs:{(lo+hi)/2},5.7) {{Rank interval}};' + '\n')
    MANIFEST[name] = []
    for i, method in enumerate(METHODS):
        vals = [r[method+'_coverage'] for r in rows]
        point = (float(np.mean(vals)), 5-i, float(np.std(vals)))
        curves += line(method, [point], horizontal=True)
        MANIFEST[name].append(dict(panel='a', method=method, points=[point]))
    labels = ','.join('{'+n+'}' for n in NAMES[::-1])
    body += axis(129, 202, 331, 102,
                 'xmin=.7,xmax=1.02,ymin=-.6,ymax=6,xtick={.7,.8,.9,1},ytick={0,1,2,3,4,5},'
                 + 'yticklabels={'+labels+r'},xlabel={Clean-label coverage},ymajorgrids=false', curves)
    body += panel(42, 162, '(b) Suspect recall')
    curves = r'\addplot[reference,dashed,line width=.6pt] coordinates {(0,.8) (1,.8)};' + '\n'
    rows = RESULTS['exp4_flag_recall']
    for method in ['uniform_shift','sharp_suspect','naive']:
        points = aggregate(rows, method+'_coverage', 'recall_target', [0,.25,.5,.75,1])
        curves += line(method, points)
        MANIFEST[name].append(dict(panel='b', method=method, points=points))
    body += axis(42, 57, 175, 90,
                 r'xmin=-.04,xmax=1.04,ymin=.72,ymax=.83,xtick={0,.5,1},xticklabels={0\%,50\%,100\%},'
                 r'ytick={.74,.78,.82},ylabel={Coverage},xlabel={True errors flagged}', curves)
    body += panel(302, 162, '(c) Sharp count bounds')
    sharp = RESULTS['exp5_sharpness']
    curves = ''
    for y, key, color in [(2,'lower_attained','navy'),(1,'upper_attained','slate')]:
        value = sharp[key]
        curves += (fr'\addplot[xbar,bar width=11pt,bar shift=0pt,fill={color},draw=none] coordinates {{({value},{y})}};' + '\n'
                   + fr'\node[text=white] at (axis cs:{value/2},{y}) {{{value}/{sharp["trials"]}}};' + '\n')
    MANIFEST[name].append(dict(panel='c', lower=sharp['lower_attained'], upper=sharp['upper_attained'], trials=sharp['trials']))
    body += axis(302, 57, 158, 90,
                 r'xmin=0,xmax=210,ymin=.4,ymax=2.6,xtick={0,100,200},ytick={1,2},'
                 r'yticklabels={Upper $U(q)$,Lower $L(q)$},ymajorgrids=false,xmajorgrids=true,xlabel={Thresholds attained}', curves)
    body += legend(y=10, methods=['uniform_shift','sharp_suspect','naive'],
                   names=['Rank shift','Exact (partial flags)','Naive CP'])
    return name, wrap(body, 338)


def audit_comparison():
    name = 'fig_revision_audit'
    body = ''
    MANIFEST[name] = []
    for i, field in enumerate(['sharp_suspect_coverage','sharp_suspect_size']):
        curves = ''
        if i == 0:
            curves += r'\addplot[reference,dashed,line width=.6pt] coordinates {(-.45,.9) (1.45,.9)};'+'\n'
        for x, (arm, alpha, color, mark) in enumerate([
                ('audit_pipeline',.05,'navy','*'), ('random_audit_fixed_cp',.1,'teal','square*')]):
            rows = sorted([r for r in REVISION['audit_30seeds'] if r['audit']==arm and r['alpha']==alpha], key=lambda r:r['seed'])
            if arm == 'audit_pipeline':
                metric = 'coverage' if i == 0 else 'avg_set_size'
                vals = np.array([r['exact'][metric] for r in REVIEW['audit_frontier'] if r['r'] == 200 and r['gamma'] == .01])
            else:
                vals = np.array([r[field] for r in rows])
            coords = ' '.join(f'({number(xx)},{number(v)})' for xx, v in zip(x+np.linspace(-.065,.065,len(vals)), vals))
            curves += fr'\addplot[only marks,mark={mark},mark size=1.5pt,draw={color},fill={color},opacity=.48] coordinates {{{coords}}};'+'\n'
            mu, sd = float(vals.mean()), float(vals.std(ddof=1))
            curves += (r'\addplot+[only marks,color=ink,mark=diamond*,mark size=2pt,mark options={solid,fill=ink,draw=ink},'
                       r'error bars/.cd,y dir=both,y explicit,error bar style={line width=1pt},error mark options={rotate=90,mark size=4pt}] '
                       + fr'coordinates {{({x},{number(mu)}) +- (0,{number(sd)})}};'+'\n')
            MANIFEST[name].append(dict(panel=i, arm=arm, n=len(vals), mean=mu, sample_sd=sd, values=vals.tolist()))
        opts = (r'xmin=-.45,xmax=1.45,xtick={0,1},xticklabels={{Robust\\951--1047 labels},{Clean audit\\1000 labels}},'
                r'xticklabel style={align=center},')
        opts += ('ymin=.86,ymax=1,ytick={.86,.9,.94,.98,1},ylabel={Clean-label coverage}' if i==0 else
                 'ymin=1.15,ymax=1.9,ytick={1.2,1.4,1.6,1.8},ylabel={Mean set size}')
        body += axis([42,282][i], 43, 178, 130, opts, curves)
        body += panel([42,282][i], 190, ['(a) Coverage','(b) Set size'][i])
    return name, wrap(body, 204)


def mechanism():
    body = r'''
\tikzset{stage/.style={draw=slate!60,line width=.55pt,rounded corners=2pt,
 minimum width=103pt,minimum height=62pt,text width=95pt,align=center,inner sep=4pt},
 flow/.style={-{Stealth[length=4pt,width=3pt]},draw=slate,line width=.8pt}}
\node[stage] (input) at (55pt,83pt) {\textbf{1. Calibration data}\\[4pt]
 Scores $s_i(y)$; labels $\tilde y$\\Suspects $F$; budget $m$};
\node[stage,fill=paperblue,draw=navy!65] (count) at (175pt,83pt) {\textbf{2. Score bounds}\\[4pt]
 Observed scores $a_i$\\Candidate maxima $u_i$};
\node[stage,fill=paperblue,draw=navy!65] (sweep) at (295pt,83pt) {\textbf{3. Two quantiles}\\[4pt]
 Select two ranks\\Return their minimum$};
\node[stage] (output) at (415pt,83pt) {\textbf{4. Prediction set}\\[4pt]
 $C_{\hat q}(x)=$\\$\{y:s(x,y)\le\hat q\}$};
\draw[flow] (input.east)--(count.west);
\draw[flow] (count.east)--(sweep.west);
\draw[flow] (sweep.east)--(output.west);
\node[text=navy] at (235pt,36pt) {$L(q)=\tilde G(q)-\min\{m,|\mathcal S(q)|\}$};
\node[text=ink,align=center] at (235pt,13pt) {Validity requires every actual error to lie in $F$ and their number to be at most $m$.};
'''
    return 'fig_method_overview', wrap(body, 122)


def runtime():
    name = 'fig_runtime_scaling'
    source = json.loads((ROOT / 'results/benchmark.json').read_text())
    curves = ''
    MANIFEST[name] = []
    methods = [('Two quantiles','decomposed_s','navy'),('Greedy search','greedy_binary_s','teal'),('Sweep','sweep_s','slate')]
    for fraction, dash, mark in [(.1,'solid','*'),(.5,'dashed','square*')]:
        for method, field, color in methods:
            points=[]
            for n in [100,1000,10000,100000,1000000]:
                vals=[r[field] for r in source['paired_scaling'] if r['n']==n and r['flag_fraction']==fraction]
                med=float(np.median(vals));points.append((n,med,med-min(vals),max(vals)-med))
            coords=' '.join(fr'({n},{number(mu)}) += (0,{number(hi)}) -= (0,{number(lo)})' for n,mu,lo,hi in points)
            curves += (fr'\addplot+[color={color},{dash},mark={mark},mark size=2pt,mark options={{solid,fill={color},draw={color}}},line width=1pt,'
                       + r'error bars/.cd,y dir=both,y explicit,error bar style={solid,line width=.6pt},error mark options={rotate=90,mark size=2pt}] coordinates {'+coords+'};\n')
            MANIFEST[name].append(dict(method=method,flag_fraction=fraction,points=points))
    body=axis(50,65,405,118,r'xmode=log,ymode=log,xmin=75,xmax=1300000,ymin=.000005,ymax=5,xtick={100,10000,1000000},ytick={.00001,.001,.1,1},xlabel={Calibration size $n$},ylabel={Wall time (s)},log basis x=10,log basis y=10',curves)
    for j,(fraction,dash,mark) in enumerate([(.1,'solid','*'),(.5,'dashed','square*')]):
        for i,(method,field,color) in enumerate(methods):
            x=[50,195,340][i];y=[21,6][j]
            body += (fr'\draw[color={color},{dash},line width=1pt] ({x}pt,{y}pt)--({x+15}pt,{y}pt);'
                     + '\n' + fr'\draw[color={color},mark={mark},mark size=2pt] plot coordinates {{({x+7.5}pt,{y}pt)}};'
                     + '\n' + fr'\node[anchor=west,text=ink] at ({x+20}pt,{y}pt) {{{method}, $f={fraction}$}};'+'\n')
    return name,wrap(body,196)


def build(name, source):
    path = TEX / (name+'.tex')
    path.write_text(source)
    result = subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',path.name],
                            cwd=TEX, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (TEX / (name+'.build.log')).write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(result.stdout[-3500:])
    (OUT / (name+'.pdf')).write_bytes((TEX / (name+'.pdf')).read_bytes())
    subprocess.run(['pdftoppm','-scale-to','1800','-png','-singlefile',str(OUT/(name+'.pdf')),str(OUT/name)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('Built',name,flush=True)


def main():
    for function in [reliability, efficiency, noise_sweep, diagnostics, audit_comparison, mechanism, runtime]:
        build(*function())
    (OUT/'pgf_data_manifest.json').write_text(json.dumps(MANIFEST,indent=2)+'\n')


if __name__ == '__main__':
    main()
