import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
from method import decomposed_quantile, greedy_binary_quantile, sharp_suspect_quantile
from validate import generated


def main():
    rng = np.random.default_rng(20261024)
    checks = []
    for trial in range(2000):
        n = int(rng.integers(1, 120)); classes = int(rng.integers(1, 12))
        S = rng.integers(0, 20, (n, classes)) / 19
        y = rng.integers(classes, size=n); flags = rng.random(n) < rng.uniform()
        m = int(rng.integers(n+1)); alpha = float(rng.uniform(.005, .99))
        q = decomposed_quantile(S, y, flags, m, alpha)
        assert q == sharp_suspect_quantile(S, y, flags, m, alpha)
        checks.append(dict(trial=trial, n=n, classes=classes, agree=True))
    rows = []
    methods = [('decomposed', decomposed_quantile), ('greedy_binary', greedy_binary_quantile), ('sweep', sharp_suspect_quantile)]
    for f, budget in [(.1, .01), (.5, .05)]:
        for n in [100, 1000, 10000, 100000, 1000000]:
            for rep in range(5):
                args = generated(rng, n, 10, f, budget)
                row = dict(n=n, classes=10, flag_fraction=f, rep=rep)
                qs=[]
                order=methods[rep%3:]+methods[:rep%3]
                for name, method in order:
                    start=time.perf_counter();q=method(*args,.1);row[name+'_s']=time.perf_counter()-start;qs.append(q)
                assert qs[0]==qs[1]==qs[2]
                row['agree']=True;rows.append(row)
            print('paired benchmark', f, n, flush=True)
    output=Path('results/benchmark.json')
    output.write_text(json.dumps(dict(environment=dict(python=platform.python_version(),machine=platform.machine(),numpy=np.__version__),random_identity_checks=checks,paired_scaling=rows,
        protocol='Fixed seed 20261024; identical generated instances; five repetitions per configuration; method order rotated between repetitions.'),indent=1)+'\n')
    print('Saved',output,flush=True)


if __name__=='__main__':
    main()
