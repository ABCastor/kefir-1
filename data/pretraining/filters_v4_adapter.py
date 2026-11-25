"""Builder adapter for filters_v4: the --filter-module interface of build_raw_parallel.py.

    python build_raw_parallel.py --filter-module filters_v4_adapter ...

Exposes exactly what load_filter() requires:
  VERSION            recorded in contract.json
  normalized         re-exported from the pinned filters.py (v1) so exact-dedup hashes and the
                     validation-overlap guard stay byte-identical; never re-implemented here
  ProseFilter        .assess(text) -> (reason | None, info)
  reject_reason      second pass, deliberately a no-op (see below)

Decision policy (assess), in this order:
  1. classify_v4(text): 'reject'     -> reason 'v4_reject:<top_axis>'      (excluded)
                        'borderline' -> reason 'v4_quarantine:<top_axis>'  (excluded, but the
                                        receipt counter is distinct, so quarantined ids can be
                                        recovered later from the source stream)
  2. language, only for documents v4 accepts (fastText lid.176.bin passed by the builder):
     chunks of 2,000 chars, newlines flattened, P(en) per chunk; the document passes when the
     word-weighted mean P(en) >= LANG_MEAN_MIN (0.65) AND at most LANG_LOW_SHARE_MAX (30%) of
     the chunks have P(en) < LANG_LOW (0.5). Otherwise reason 'language_confidence'. A model
     exception -> 'language_error' (excluded, never silently accepted); no scorable chunk ->
     'language_unverified'. This replaces v1's 0.90-on-every-chunk veto, which rejected 46/400
     random documents of which an external judge called 7/12 sampled ones clear English.
  3. student gate (student_gate.StudentGate, distilled from a judge's labels on 16k random docs), only
     for documents that passed 1 and 2: 'student_reject:<target>' when a gated target reaches its
     threshold (precision-first thresholds live in student_gate.THRESHOLDS). Reasoning targets are
     never gated here: they are stored as tags for the recipe/eye review.
  4. accept: reason None, info = v4_bad_score, v4_decision, v4_top_axes (3), v4_english_mean,
     v4_english_min, v4_chunks, st_<target> probabilities. Prefixes 'v4_'/'st_' are not builder-reserved.

Second pass: reject_reason() always returns None. The pinned refine_seed.reject_reason is four
single-hit vetoes (function_notation, inline_assignment, numeric_formula, ascii_symbolic_logic).
On the 400-document random sample it would reject 8 documents: 2 that v4 already rejects or
quarantines, and 6 ordinary prose such as "a 128 × 48 pixel image", "the .40 S&W cartridge" and
"N= 12×300 = 3600 pairs". v4 scores exactly those shapes as weak evidence (math_inline,
math_unicode, equation lines) instead of vetoing on one hit, so the second pass is disabled by
exposing this no-op; contract.json then records second_pass = 'filters_v4_adapter.reject_reason'.
"""
from normalization import normalized  # noqa: F401  exact v1 normalization
from filters_v4 import classify_v4, VERSION as _V4_VERSION, WORDS

from student_gate import StudentGate, STUDENT_PATH

VERSION = _V4_VERSION + '+adapter-2'   # adapter-2: student gate after v4 + language
RESERVED = frozenset(['text', 'id', 'source_revision', 'source_file', 'record_start', 'cluster_id',
                      'source_token_count', 'target_token_count', 'token_offset_in_shard'])
CHUNK_CHARS = 2000
MIN_CHUNK_WORDS = 5
LANG_MEAN_MIN = 0.65
LANG_LOW = 0.50
LANG_LOW_SHARE_MAX = 0.30
EN_LABEL = '__label__en'


def top_axes(axes, k=3):
    return [name for name, _ in sorted(axes.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def english_scores(language_model, text):
    """[(P(en), words)] per chunk. P(en) is 0 when 'en' is not among the returned labels."""
    out = []
    for i in range(0, len(text), CHUNK_CHARS):
        chunk = text[i:i + CHUNK_CHARS].replace('\n', ' ').replace('\r', ' ')
        words = len(WORDS.findall(chunk))
        if words < MIN_CHUNK_WORDS:
            continue
        labels, probs = language_model.predict(chunk, k=-1)
        p = 0.0
        for label, prob in zip(labels, probs):
            if label == EN_LABEL:
                p = float(prob)
                break
        out.append((p, words))
    return out


def language_verdict(scores):
    """(reason | None, mean, min). Word-weighted mean >= 0.65 and <= 30% of chunks below 0.5."""
    if not scores:
        return 'language_unverified', None, None
    total = sum(w for _, w in scores)
    mean = sum(p * w for p, w in scores) / total
    low_share = sum(1 for p, _ in scores if p < LANG_LOW) / len(scores)
    lowest = min(p for p, _ in scores)
    if mean < LANG_MEAN_MIN or low_share > LANG_LOW_SHARE_MAX:
        return 'language_confidence', mean, lowest
    return None, mean, lowest


class ProseFilter:
    def __init__(self, language_model, student=None):
        self.language_model = language_model
        self.student = StudentGate(STUDENT_PATH) if student is None else student

    def assess(self, text):
        if not isinstance(text, str):
            return 'not_text', {}
        r = classify_v4(text)
        axes = top_axes(r['axes'])
        if r['decision'] == 'reject':
            return 'v4_reject:' + axes[0], {}
        if r['decision'] == 'borderline':
            return 'v4_quarantine:' + axes[0], {}
        try:
            scores = english_scores(self.language_model, text)
        except Exception:
            return 'language_error', {}
        reason, mean, lowest = language_verdict(scores)
        if reason:
            return reason, {}
        if self.student is not False:
            reason, st = self.student.verdict(text)
            if reason:
                return reason, {}
        else:
            st = {}
        info = {'v4_bad_score': r['bad_score'], 'v4_decision': r['decision'], 'v4_top_axes': axes,
                'v4_english_mean': round(mean, 4), 'v4_english_min': round(lowest, 4),
                'v4_chunks': len(scores)}
        info.update({'st_' + k: round(v, 4) for k, v in st.items()})
        assert not (set(info) & RESERVED)
        return None, info


def reject_reason(text):
    """Second pass disabled: v4 already scores these shapes; see module docstring."""
    return None
