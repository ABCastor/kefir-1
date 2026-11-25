"""Student gate: cheap linear classifiers distilled from a judge's labels (trained in the original,
unpublished labelling workspace).

Loaded once per worker from STUDENT_PATH (a pickle of {name: sklearn Pipeline}); each pipeline is
hashing char/word n-grams -> tf-idf -> logistic regression, so inference is a few ms per document and
needs only numpy/scipy/scikit-learn at the versions pinned in requirements.txt.

Targets (probability that the judge would say so):
  poor_english     the English is poor: spun, machine-translated, garbled, truncated
  not_prose        not connected English prose: lists, menus, tables, keyword pages, other languages
  reasoning_dense  the text is largely verbal argument/explanation (because, therefore, if-then)
  reasoning_none   the text states/lists/narrates/advertises without reasoning

Thresholds are chosen precision-first on the held-out split (source is abundant; a false reject
costs nothing, a false accept is the failure) and recorded in THRESHOLDS with their provenance.
"""
import hashlib
import pickle
from pathlib import Path

STUDENT_PATH = Path(__file__).resolve().parent / 'assets' / 'students.pkl'
CAP = 4000                       # default; the pickle carries the CAP it was trained with
THRESHOLDS = {                   # from the student training report (16k labels, 2026-09-20); None = tag only
    'poor_english': 0.80,        # precision 0.81 vs judge, flags 2.1% of the stream, recall 0.16
    'not_prose': 0.80,           # precision 0.85, flags 1.1%, recall 0.16
    'symbolic': 0.70,            # precision 0.73, flags 0.5%; v4 already catches the dense cases
}


STUDENT_SHA256 = 'b8b8472278bc63a04f124b66fbefff9972a30d39ae98b13b541b772874135959'


def require_student_asset(path=STUDENT_PATH):
    """Return the student pickle path, or explain how to obtain it.

    The pickle is not distributed with this repository: its classifiers were trained on
    labels whose terms (CC BY-NC 4.0) are not resolved for an MIT tree. Obtain the exact
    file separately and place it at this path; its SHA-256 is checked here, before the pickle
    is ever opened, and a file with any other hash is refused.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f'missing student classifier asset {path}: it is not distributed with this '
            f'repository; obtain students.pkl separately (SHA-256 {STUDENT_SHA256}) and '
            'place it there before building the corpus')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != STUDENT_SHA256:
        raise RuntimeError(
            f'student classifier asset {path} has SHA-256 {digest.hexdigest()}, expected '
            f'{STUDENT_SHA256}; refusing to unpickle it')
    return path


class StudentGate:
    def __init__(self, path=STUDENT_PATH, thresholds=None):
        with open(require_student_asset(path), 'rb') as f:
            bundle = pickle.load(f)
        self.features = bundle['features']       # shared hashing + tf-idf, applied once per document
        self.cap = bundle.get('cap', CAP)
        self.models = bundle['models']
        self.thresholds = dict(THRESHOLDS if thresholds is None else thresholds)
        self.names = sorted(self.models)

    def scores(self, text):
        x = self.features.transform([text[:self.cap]])
        return {name: float(self.models[name].predict_proba(x)[0, 1]) for name in self.names}

    def verdict(self, text):
        """(reason | None, scores). reason = 'student_reject:<target>' for the first gated target
        whose probability reaches its threshold, in THRESHOLDS order."""
        s = self.scores(text)
        for name, thr in self.thresholds.items():
            if thr is not None and name in s and s[name] >= thr:
                return 'student_reject:' + name, s
        return None, s
