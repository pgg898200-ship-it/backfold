"""Time an indexed Laplace head and measure actual preview correction density."""
import concurrent.futures
import json
import platform
import statistics
import time
import warnings
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

def clock(fn, n):
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter()-start)*1000)
    return statistics.median(samples)

def make_index(keys, values):
    order = np.argsort(keys,kind="stable")
    k = keys[order]
    v = np.column_stack((values[order],np.ones(len(k))))
    forward, backward = v.copy(), v.copy()
    decay = np.exp(-np.diff(k))
    for i,d in enumerate(decay):
        forward[i+1] += d*forward[i]
    for i in range(len(decay)-1,-1,-1):
        backward[i] += decay[i]*backward[i+1]
    return k, forward, backward

def ask(index, q):
    keys, prefix, suffix = index
    boundary = np.searchsorted(keys,q,side="right")
    left, right = np.maximum(boundary-1,0),np.minimum(boundary,len(keys)-1)
    dl = np.where(boundary>0,np.abs(q-keys[left]),np.inf)
    dr = np.where(boundary<len(keys),np.abs(q-keys[right]),np.inf)
    nearest = np.minimum(dl,dr)
    out = np.exp(nearest-dl)[:,None]*prefix[left]
    out += np.exp(nearest-dr)[:,None]*suffix[right]
    return out[:,:-1]/out[:,-1:]

def attention():
    rng = np.random.default_rng(0)
    k = rng.uniform(-12,12,32768)
    k[:5] = 0.0
    v = rng.normal(size=(len(k),32))
    q = np.r_[rng.uniform(-12,12,1020),-14.0,0.0,0.0,14.0]
    def direct():
        out = []
        for chunk in np.array_split(q,32):
            weights = np.exp(-np.abs(chunk[:,None]-k))
            out.extend((weights@v)/weights.sum(axis=1,keepdims=True))
        return np.array(out)
    index = make_index(k,v)
    exact = direct()
    maximum = float(np.max(np.abs(exact-ask(index,q))))
    assert maximum < 1e-10
    return dict(keys=len(k),queries=len(q),channels=v.shape[1],setup_ms=clock(lambda:make_index(k,v),5),
                indexed_ms=clock(lambda:ask(index,q),31),dense_ms=clock(direct,5),max_abs_error=maximum,
                index_bytes=sum(a.nbytes for a in index),input_bytes=k.nbytes+v.nbytes)

def preview():
    x, y = load_digits(return_X_y=True)
    a,b,c,d = train_test_split(x/16,y,test_size=.25,random_state=7,stratify=y)
    model = MLPClassifier(hidden_layer_sizes=(256,),max_iter=100,random_state=19)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",ConvergenceWarning)
        model.fit(a,c)
    w1,w2 = model.coefs_
    b1,b2 = model.intercepts_
    true = np.maximum(0,b@w1+b1)
    logits = true@w2+b2
    results = []
    for bits in [1,2,3,4]:
        coarse = np.round(b*(2**bits-1))/(2**bits-1)
        guess = np.maximum(0,coarse@w1+b1)
        delta = true-guess
        result = guess@w2+delta@w2+b2
        results.append(dict(bits=bits,nonzero_fraction=float(np.count_nonzero(delta)/delta.size),
                            max_logit_error=float(np.max(np.abs(result-logits))),
                            preview_accuracy=float(np.mean(np.argmax(guess@w2+b2,axis=1)==d))))
    return dict(dataset="sklearn digits",test_images=len(d),model_accuracy=float(model.score(b,d)),rows=results)

def overlap():
    rng = np.random.default_rng(9)
    n = 1536
    a, w = [np.asfortranarray(rng.normal(size=(n,n)).astype("f")) for _ in range(2)]
    x = rng.normal(size=n).astype("f")
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        for k in (16,192,1536):
            def direct():
                delta = (a@x)[:k]
                actual = x.copy()
                actual[:k] += delta
                return w@actual
            def ahead():
                task = pool.submit(np.matmul,w,x)
                delta = (a@x)[:k]
                return task.result()+w[:,:k]@delta
            truth = direct()
            rel = float(np.linalg.norm(ahead()-truth)/np.linalg.norm(truth))
            assert rel < 1e-4
            hits = [[],[]]
            for i in range(64):
                for j in (i%2,1-i%2):
                    fn = (direct,ahead)[j]
                    hits[j].append(clock(fn,1))
            rows.append(dict(width=n,active=k,serial_ms=statistics.median(hits[0]),
                             parallel_ms=statistics.median(hits[1]),relative_error=rel))
    return rows
if __name__ == "__main__":
    import sys
    with threadpool_limits(limits=1):
        result = dict(platform=platform.machine(),attention=attention(),early_layer=preview(),constructed_overlap=overlap())
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+"\n")
