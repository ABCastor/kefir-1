"""English-prose screening v4: a SCORE with many weak flags, never a single-keyword veto.

Whole-document decisions only; the text is never edited. The product is ``bad_score`` in [0, 1]
plus per-axis flags and coverage counts; ``decision`` is only a thresholding of the score.

Design (see plan 08 §3 and §7.3):
- Code evidence is LINE-SHAPED (indentation + statement shape, fenced blocks, operator digraphs,
  data-structure lines, shell/SQL statement lines) and counts BOTH lines and character coverage.
  A document is "dense code" only above both. ``for`` is an English word; ``Import duties`` is prose.
- Math evidence: LaTeX commands with coverage, ``$...$`` spans only when the span is math-like
  (``$12 ... $30`` is currency; ``$x$`` is math), Unicode math operators and ISOLATED Greek letters
  by count AND density per 1,000 chars, equation-like lines by line share. Superscript footnote
  markers (¹²³) never count. Every count-based flag needs >= 2 occurrences: one hit is never evidence.
- Equation soup (extraction of a maths PDF): windows of whitespace tokens where >= 40% are short
  letter+digit symbols (D8, 2D6, S00), lone letters or bare operators, plus numbered-equation
  markers like ``(6.10)``; windows AND token share, so ``30 x 30m`` or ``vitamin B12`` never fire.
- HTML tag mass is an EXTRACTION axis, not code. URLs and e-mails are counted flags, not vetoes.
- No domain word-list. "formal logic", "predicate calculus", "programming" score 0 on their own.
- Structural quality flags (word count, letter density, non-ASCII letters, repeated lines,
  top-trigram share, ...) are soft scores with small weights.

Combination rule (monotone, documented in ``combine``):
    hard      = max(code, math, soup, html)                each family in [0, 1]
    soft      = min(SOFT_CAP, sum_i weight_i * flag_i)     each flag in [0, 1]
    bad_score = min(1, hard + soft)
    decision  = accept if bad < 0.25, borderline if bad < 0.6, reject otherwise.
Inside a family the sub-scores are combined as ``max + 0.5 * (sum - max)`` capped at 1, so more
independent evidence raises the family score but correlated evidence cannot double count fully.

Language identification is out of scope (handled elsewhere). Pure Python + regex, no ML.
"""
import re
from collections import Counter

VERSION = 'english-prose-v4-scored-r2'  # r2: script family (foreign letters, fancy alphanumerics)
MAX_CHARS = 200_000

# --------------------------------------------------------------------------------------------
# All weights and thresholds. ``ramp(x, lo, hi)`` is 0 at/below lo, 1 at/above hi, linear between.
# --------------------------------------------------------------------------------------------
PARAMS = {
    'thresholds': {'accept': 0.25, 'reject': 0.60},
    'min_words': 40,
    'soft_cap': 1.0,
    # ---- code family (hard) ----
    'code_lines': (2, 8),            # ramp on number of line-shaped code lines
    'code_cov': (0.10, 0.40),        # ramp on char coverage of those lines (of the detection view)
    'fence_cov': (0.05, 0.40),       # ramp on chars inside ``` / ~~~ fences
    'digraph_lines': (2, 10),        # lines containing ==, !=, &&, ||, =>, ->, +=, -=, ::, :=
    'digraph_cov': (0.05, 0.35),
    'symbol_mass': (0.04, 0.16),     # share of {}[]();=<>|&^~#@$\ chars
    'json_pairs': (6, 30),           # "key": pairs, count AND density: serialized JSON on one long line
    'json_pairs_per_1k': (3.0, 12.0),
    'bibtex_fields': (2, 6),         # BibTeX-style `title = {` / `author = "` field lines; count only (a rare shape)
    # ---- math family (hard) ----
    'latex_cmds': (2, 15),           # \command count
    'latex_cov': (0.03, 0.25),       # coverage of \command{...}, $$..$$, \[..\], equation envs
    'dollar_spans': (2, 12),         # $...$ spans with math-like content
    'dollar_cov': (0.03, 0.25),
    'unicode_math_count': (2, 15),        # count AND density: a single α is never evidence
    'unicode_math_per_1k': (3.0, 20.0),   # math operators + isolated Greek per 1,000 chars
    'equation_lines': (1, 6),        # equation-like lines (x = y + z)
    'equation_share': (0.10, 0.40),  # share of non-blank lines that are equation-like
    'inline_eq_count': (2, 12),      # count AND density: one `F = ma` is never evidence
    'inline_eq_per_1k': (1.5, 8.0),  # inline `x = y`, `f(x)`, `x^2` per 1,000 chars
    # ---- equation-soup family (hard): broken maths from PDF extraction ----
    'soup_windows': (1, 5),          # 16-token windows with >=40% soup tokens and >=2 letter+digit symbols
    'soup_share': (0.04, 0.20),      # share of whitespace tokens inside such windows
    'eq_markers': (2, 6),            # own-line markers, plus inline ones only when soup windows exist
    'eq_markers_per_1k': (0.5, 3.0),
    # ---- script family (hard): the corpus is English-only, tokenizer has no CJK/Cyrillic tokens ----
    'foreign_letters': (3, 8),       # letters from non-Latin, non-Greek scripts (CJK, kana, Hangul, Cyrillic, Arabic, ...)
    'fancy_alnum': (4, 20),          # mathematical/enclosed/fullwidth alphanumerics (𝑌𝑚, ①, Ｆ): decoration or extracted maths
    # ---- html / extraction family (hard) ----
    'tag_count': (2, 20),
    'tag_mass': (0.08, 0.45),
    'entity_per_1k': (2.0, 15.0),
    # ---- soft flags: (ramp lo, ramp hi, weight) ----
    'short_doc': (0.0, 1.0, 0.60),        # x = 1 - words/min_words
    'letter_density': (0.0, 0.25, 0.60),  # x = 0.60 - letters/chars
    'non_ascii_letters': (0.02, 0.15, 0.40),
    'digit_share': (0.20, 0.50, 0.50),    # digits / alnum
    'repeated_lines': (0.10, 0.50, 0.60), # char mass of duplicated lines (>=30 chars)
    'top_trigram': (0.08, 0.30, 0.60),    # 3*count(top trigram)/words
    'top_word': (0.15, 0.35, 0.40),
    'short_lines': (0.50, 0.90, 0.50),    # share of non-blank lines with < 4 words (>= 8 lines); alone -> borderline, never reject
    'url_count': (3, 20, 0.40),
    'url_mass': (0.05, 0.30, 0.40),
    'email_count': (1, 10, 0.15),
    'control_chars': (1, 20, 0.50),       # C0 controls (not \n\r\t) and U+FFFD
    'special_tokens': (1, 3, 0.50),       # <|...|> literals
    'mojibake_per_1k': (1.0, 8.0, 0.40),  # Ã©, â€, Â  ...
}

# --------------------------------------------------------------------------------------------
# Regexes. Case-SENSITIVE on purpose: code keywords are lowercase, prose sentences start capitalized.
# --------------------------------------------------------------------------------------------
WORDS = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
NON_ASCII = re.compile(r'[^\x00-\x7f]')
# Serialized JSON on a single long line has no line shape: count `"key":` pairs by density instead.
JSON_PAIR = re.compile(r'"[A-Za-z_][A-Za-z0-9_.-]{0,40}"\s*:\s*(?=["\[{tfn0-9-])')
# BibTeX / citation-export field lines: `title = {...}`, `author = "..."`, `year = 2014,`
BIBTEX_FIELD = re.compile(r'(?im)^\s*(?:title|author|journal|booktitle|publisher|year|volume|number|pages|doi|url|abstract|keywords|issn|isbn|month|editor|address|note|howpublished)\s*=\s*[{"\d]')
# Letters of scripts that are never English: Cyrillic..Georgian, Indic, Thai/Lao/Tibetan/Myanmar,
# Ethiopic, Khmer, CJK radicals..Yi (incl. Hiragana/Katakana/Bopomofo), Hangul, CJK compatibility,
# Arabic presentation forms, halfwidth Katakana, supplementary CJK. Greek, Latin extensions and IPA
# are deliberately NOT here (Greek is a maths signal, IPA appears in English dictionary entries).
FOREIGN_SCRIPT = re.compile(r'[\u0400-\u052f\u0530-\u058f\u0590-\u05ff\u0600-\u06ff\u0750-\u077f'
                            r'\u0900-\u0dff\u0e00-\u0eff\u0f00-\u0fff\u1000-\u109f\u10a0-\u10ff'
                            r'\u1100-\u11ff\u1200-\u137f\u1780-\u17ff\u2e80-\u2fdf\u3041-\u30ff'
                            r'\u3105-\u312f\u3131-\u318e\u31a0-\u31bf\u31f0-\u31ff\u3400-\u4dbf'
                            r'\u4e00-\u9fff\ua000-\ua4cf\uac00-\ud7af\uf900-\ufaff\ufb50-\ufdff'
                            r'\ufe70-\ufeff\uff66-\uff9f\U00020000-\U0002fa1f]')
# Alphanumerics that only appear as decoration or as extracted maths: Mathematical Alphanumeric
# Symbols, Enclosed Alphanumerics (+ Supplement), fullwidth Latin letters.
FANCY_ALNUM = re.compile(r'[\U0001d400-\U0001d7ff\u2460-\u24ff\U0001f100-\U0001f1ff\uff21-\uff3a\uff41-\uff5a]')
SPECIAL_TOKEN = re.compile(r'<\|[^|<>\n]{1,40}\|>')
MOJIBAKE = re.compile(r'Ã[\x80-\xbf¡-¿À-ÿ]|â€|Â[ £©®°]')
TAG = re.compile(r'</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]{0,300})?/?>|<!--.*?-->|<!DOCTYPE[^>]*>', re.S)
ENTITY = re.compile(r'&(?:[a-zA-Z]{2,8}|#\d{1,6}|#x[0-9a-fA-F]{1,5});')
URL = re.compile(r'(?:https?://|www\.)[^\s<>"\')\]]+')
EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
FENCE = re.compile(r'(?m)^[ \t]*(```|~~~)[^\n]*\n.*?^[ \t]*\1[ \t]*$', re.S)
SYMBOLS = re.compile(r'[{}\[\]();=<>|&^~#@$\\]')

# One multiline alternation: every branch matches a WHOLE line that has code shape.
# Performance rule: never write `[^\n]*A[^\n]*B$` (quadratic backtracking on long paragraphs);
# express "contains A and ends with B" as anchored lookaheads, each O(line length).
_KW_INDENT = (
    # Python-style statements (plain keyword is enough), typed/modifier statements (need code
    # punctuation in the tail: "  string name;" yes, "  string of pearls" no), and bare block closers.
    r'(?:def\s+\w+\s*\(|class\s+\w+\s*[(:]|import\s+[\w.]+|from\s+[\w.]+\s+import\b'
    r'|(?:if|elif|for|while|with|except|try|else|finally)\b(?=[^\n]*:[ \t]*$)'
    r'|return\b|yield\b|raise\s+\w|break[ \t]*;?[ \t]*$|continue[ \t]*;?[ \t]*$|pass[ \t]*$'
    r'|(?:public|private|protected|static|final|abstract|export|async|function|const|let|var'
    r'|int|void|float|double|char|bool|string|auto|struct|typedef|enum|template|virtual|override'
    r'|fn|pub|impl|use|local|elseif)\b(?=[^\n]*[;{}()=\[\]<>])'
    r'|(?:end|fi|done|then|do|esac|endif|EOF)[ \t]*;?[ \t]*$)'
)
_TYPED_KW = (
    r'(?:public|private|protected|static|final|abstract|export|async|function|const|let|var'
    r'|int|void|float|double|char|bool|string|auto|struct|typedef|enum|template|virtual'
    r'|fn|pub|impl|use)\b'
)
_SQL_KW = (
    r'(?:SELECT|INSERT\s+INTO|CREATE\s+(?:TABLE|INDEX|DATABASE|VIEW)|DROP\s+(?:TABLE|DATABASE)'
    r'|UPDATE|DELETE\s+FROM|ALTER\s+TABLE|VALUES|LOCK\s+TABLES|UNLOCK\s+TABLES|WHERE|FROM|INNER\s+JOIN'
    r'|LEFT\s+JOIN|GROUP\s+BY|ORDER\s+BY|PRIMARY\s+KEY|KEY|CONSTRAINT|ENGINE=|SET\s+\w+=|USE\s+\w+;'
    r'|BEGIN;|COMMIT;)\b'
)
_SHELL_KW = (
    r'(?:\$\s|sudo\s|export\s+\w+=|set\s+-[a-z]+'
    r'|(?:pip3?|npm|yarn|apt(?:-get)?|yum|dnf|brew|conda|cargo|go|docker|git|make)\s+'
    r'(?:install|add|update|upgrade|run|build|pull|push|clone|commit|checkout|test|get)\b'
    r'|cd\s+\S|mkdir\s+-|chmod\s+[0-7+-]|chown\s+\w|rm\s+-|wget\s|curl\s+-|echo\s+"|cat\s+\S+\s*[|>]'
    r'|grep\s+-|sed\s+-|awk\s+\'|ls\s+-|ln\s+-|find\s+\S+\s+-|mysqldump\s|tar\s+-|ssh\s+\w)'
)
# Code statements are short: fewer than 10 whitespace-separated tokens on the line.
_SHORT = r'(?![ \t]*(?:\S+[ \t]+){9,})'
CODE_LINE = re.compile(
    r'(?m)^(?:'
    # A. indented (>=2 spaces or a tab) AND a statement shape
    r'(?:[ ]{2,}|\t)[ \t]*(?:'
    + _KW_INDENT +
    r'|(?=[^\n]*:[ \t]*$)'                               # ends with ':'
    r'|(?=[^\n]*;[ \t]*$)(?=[^\n]*[=(){}\[\]])' + _SHORT +   # ends with ';' + code punctuation, few tokens
    r'|(?=[^\n]*[{}][ \t]*,?[ \t]*$)'                    # ends with { or }
    r'|(?=[^\n]*=)(?=[^\n]*[(\[{])'                      # '=' together with ( [ {
    r'|[)\]}]+[;,]?[ \t]*$'                              # closing brackets alone
    r'|"[^"\n]*"[ \t]*:'                                 # JSON "key": value
    r')'
    # B. unindented strong statement shapes (lowercase keywords: prose starts capitalized)
    r'|(?:def|class)\s+\w+\s*[(:]'
    r'|import\s+[\w.]+(?:\s*,\s*[\w.]+)*(?:\s+as\s+\w+)?[ \t]*;?[ \t]*$'
    r'|from\s+[\w.]+\s+import\s+(?:\*|[\w.]+(?:\s*,\s*[\w.]+)*)(?:\s+as\s+\w+)?[ \t]*$'
    r'|#\s*(?:include\s*[<"]|define\s+\w|ifn?def\s+\w|endif|pragma\s)'
    r'|#!\s*/'
    r'|(?:using\s+namespace|package|namespace)\s+[\w.:]+\s*;?[ \t]*$'
    r'|' + _TYPED_KW + r'(?=[^\n]*[;{)(=][ \t]*$)'
    r'|(?:if|elif|for|while|with|except|try|else|finally)\b(?=[^\n]*:[ \t]*$)'   # for x in y:
    r'|(?=[^\n]*;[ \t]*$)(?=[^\n]*[={}\[\]])' + _SHORT +    # `int x = f(a);` — not a prose list item ending in ';'
    r'|(?=[^\n]*\S)(?=[^\n]*\{[ \t]*$)'                  # line ending in {
    r'|[ \t]*[}{][ \t]*[;,]?[ \t]*$'
    r'|[ \t]*[)\]]+[;,]?[ \t]*$'
    r'|[ \t]*(?:end|fi|done|esac|endif|EOF)[ \t]*$'
    r'|[ \t]*[A-Za-z_][\w.]*(?:\[[^\]\n]*\])?\s*=\s*(?=[^=\n]*[(\[{])'   # x = foo(1)
    r'|[ \t]*[A-Za-z_]\w*=\S'                            # KEY=value (shell / ini)
    r'|[ \t]*"[^"\n]*"[ \t]*:'                           # JSON "key": value
    r'|[ \t]*[\[\]{},]+[ \t]*$'
    r'|[ \t]*' + _SQL_KW + r'(?=[^\n]*(?:[;(),`][ \t]*$|[`()]))'
    r'|[ \t]*(?:/\*!?|\*/)'                              # C / SQL block comments (`-- ` is also a prose bullet)
    r'|[ \t]*//'
    r'|[ \t]*' + _SHELL_KW +
    r'|[ \t]*(?:if\s+\[|for\s+\w+\s+in\b[^\n]*;\s*do|while\s+[^\n]*;\s*do|case\s+\S+\s+in|then|else|do)[ \t]*$'
    r'|[ \t]*\w+\(\)\s*\{[ \t]*$'
    r'|[ \t]*(?:>>>|In \[\d+\]:|Out\[\d+\]:)\s'            # REPL prompts; `... ` alone is a prose ellipsis
    r'|[ \t]*(?:at\s+[\w$.<>]+\([\w.]+:\d+\)|File "[^"\n]+", line \d+|Traceback \(most recent call last\))'
    r'|(?:\+\+\+ |--- |@@ [^\n]*@@|diff --git|index [0-9a-f]+\.\.)'
    r')[^\n]*$'
)
# `->` only when not part of `-->` / `<->` (prose arrows, WebVTT `00:01 --> 00:02` cue lines).
_DIGRAPHS = r'==|!=|&&|\|\||=>|(?<![-<])->|\+=|-=|(?<=\w)::(?=\w)|:='   # `::` only glued: std::vector
DIGRAPH_ANY = re.compile(_DIGRAPHS)
DIGRAPH_LINE = re.compile(r'(?m)^[^\n]*(?:' + _DIGRAPHS + r')[^\n]*$')
QUOTE_PREFIX = re.compile(r'(?m)^[ \t]*(?:>[ \t]?)+')      # e-mail / Usenet quoting is prose, not symbols

LATEX_CMD = re.compile(r'\\[A-Za-z]{2,}\*?')
LATEX_SPAN = re.compile(
    r'\\begin\{(equation|align|eqnarray|gather|multline|array|matrix|pmatrix|bmatrix|cases|itemize|enumerate|tabular|figure|table|abstract|document)\*?\}.*?\\end\{\1\*?\}'
    r'|\$\$.+?\$\$|\\\[.+?\\\]|\\\(.+?\\\)'
    r'|\\[A-Za-z]{2,}\*?(?:\[[^\]\n]{0,100}\])?(?:\{[^{}\n]{0,200}\})*', re.S)
DOLLAR_SPAN = re.compile(r'\$([^$\n]{1,300})\$')
MATHLIKE = re.compile(r'[\\^_{}=]|[A-Za-z0-9)]\s*[-+*/]\s*[A-Za-z0-9(]|[≤≥≠≈∑∫√∞±×÷→∈∀∃]')
CURRENCY_START = re.compile(r'^\s*\d')
# Unicode math operators, arrows used as logic, letterlike symbols; NOT ¹²³ and NOT the math-bold
# alphanumerics (𝐀-𝟿), which social-media posts use as decoration.
UNICODE_MATH = re.compile(r'[∀-⋿⟀-⟯⦀-⧿⨀-⫿ℂℇℊ-ℓℕℙ-ℝℤℵ→⇒⇔↔×÷±√∞∂∇]')
GREEK_ISOLATED = re.compile(r'(?<![A-Za-zΆ-ώ])[α-ωΑ-Ω](?![A-Za-zΆ-ώ])')
EQUATION_LINE = re.compile(
    r"(?m)^[ \t]*[A-Za-z0-9_\\(){}^.,'|+\-*/ \[\]]{1,40}?\s*(?:=|≤|≥|≠|≈|<=|>=)\s*"
    r'(?=[^\n]*(?:[-+*/^√∑∫\\≤≥≠±×÷]|\d))[^\n]{1,80}$')
# Equation soup. MIX_TOKEN is the cheap gate (a whole whitespace token of <=4 chars mixing letters
# and digits); SOUP_TOKEN classifies tokens on candidate lines only.
MIX_TOKEN = re.compile(r'(?<!\S)(?=\S{1,4}(?!\S))(?:[A-Za-z]+\d[A-Za-z0-9]*|\d+[A-Za-z][A-Za-z0-9]*)(?!\S)')
SOUP_TOKEN = re.compile(
    r'^(?:(?P<mix>(?!\d+(?:st|nd|rd|th)$)(?=.*\d)(?=.*[A-Za-z])[A-Za-z0-9]{1,4})'
    r'|(?P<op>[=+\-−*/^±×·≤≥≠≈→]\S{0,3}|[()\[\]|.,;:]|\(\d+\.\d+[a-z]?\))'
    r'|(?P<single>[B-HJ-Zb-hj-z]))$')
# Numbered-equation markers: (6.10), (3.2a). One or two decimals only, so P/E ratios "(8.724)" and
# steel grades "(1.4021)" never match; inline markers count only next to soup windows, own-line
# markers always (a lone "(6.10)" line is the displayed-equation number the PDF left behind).
EQ_MARKER = re.compile(r'\(\d{1,2}\.\d{1,2}[a-z]?\)')
EQ_MARKER_LINE = re.compile(r'(?m)^[ \t]*\(\d{1,2}\.\d{1,2}[a-z]?\)[ \t]*$')
SOUP_WINDOW = 16
SOUP_STRIDE = 8
MATH_MARK = re.compile(r'[\\^_{}=]')
EQ_MARK = re.compile(r'[=≤≥≠≈]')
# Inline-equation shapes, each anchored on its rare literal so the scan is a fast literal search;
# the "preceded by a single letter" condition is checked in Python on the few candidates.
INLINE_ASSIGN = re.compile(r'=\s*[-+]?[A-Za-z0-9(]')     # x = y, F = ma, a = -3
CARET_POWER = re.compile(r'\^\d')                        # x^2
FUNC_CALL = re.compile(r'\(\s*[a-zA-Z]\s*\)')            # f(x)
GREEK_ANY = re.compile(r'[Ͱ-Ͽ]')
MOJIBAKE_ANY = re.compile(r'[Ãâ]')
DIGIT_OR_CONTROL = re.compile(r'\d+|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+')
NGRAM_WINDOW = 2000     # words used for top-word / top-trigram shares (spam repeats early anyway)


def ramp(x, lo, hi):
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    return (x - lo) / (hi - lo)


def family(scores):
    """max + 0.5 * (sum - max), capped at 1: more independent evidence -> higher."""
    if not scores:
        return 0.0
    m = max(scores)
    return min(1.0, m + 0.5 * (sum(scores) - m))


def _ratio(a, b):
    return a / b if b else 0.0


def classify_v4(text, min_words=None, params=None):
    """Return {'bad_score', 'flags', 'coverage', 'axes', 'decision', 'filter_version'}.

    Every full-text regex scan costs ~4 ns/char, so each optional scan is gated on a cheap
    substring test for its trigger character; Unicode flags all derive from one non-ASCII pass.
    """
    p = PARAMS if params is None else {**PARAMS, **params}
    min_words = p['min_words'] if min_words is None else min_words
    if not isinstance(text, str):
        return {'bad_score': 1.0, 'flags': {'not_text': 1.0}, 'coverage': {}, 'axes': {},
                'decision': 'reject', 'filter_version': VERSION}
    n = len(text)
    if n == 0:
        return {'bad_score': 1.0, 'flags': {'empty': 1.0}, 'coverage': {'chars': 0}, 'axes': {},
                'decision': 'reject', 'filter_version': VERSION}
    if n > MAX_CHARS:
        text = text[:MAX_CHARS]
        n = MAX_CHARS
    flags, cov = {}, {'chars': n}

    # ---------------- extraction view: tags, entities, urls, emails ----------------
    tags = TAG.findall(text) if '<' in text else []
    tag_chars = sum(len(t) for t in tags)
    urls = URL.findall(text) if ('http' in text or 'www.' in text) else []
    url_chars = sum(len(u) for u in urls)
    entities = len(ENTITY.findall(text)) if '&' in text else 0
    emails = len(EMAIL.findall(text)) if '@' in text else 0
    cov.update(tag_count=len(tags), tag_chars=tag_chars, tag_cov=_ratio(tag_chars, n),
               url_count=len(urls), url_chars=url_chars, url_cov=_ratio(url_chars, n),
               entity_count=entities, email_count=emails)
    flags['html_tags'] = ramp(len(tags), *p['tag_count']) * ramp(cov['tag_cov'], *p['tag_mass'])
    flags['html_entities'] = 0.5 * ramp(entities, 2, 10) * ramp(entities * 1000.0 / n, *p['entity_per_1k'])   # ceiling 0.5
    html = family([flags['html_tags'], flags['html_entities']])

    # Detection view for code/math: tags and URLs masked (plan 08 §3), newlines preserved.
    view = text
    if tags:
        view = TAG.sub(' ', view)
    if urls:
        view = URL.sub(' ', view)
    vn = max(1, len(view))

    # ---------------- code family ----------------
    fence_chars = 0
    fences = 0
    view_nofence = view
    if '```' in text or '~~~' in text:
        spans = [m.group(0) for m in FENCE.finditer(text)]
        fences = len(spans)
        fence_chars = sum(len(f) for f in spans)
        if fence_chars:
            view_nofence = FENCE.sub(' ', view)
    code_lines = CODE_LINE.findall(view_nofence)
    code_chars = sum(len(l) for l in code_lines)
    digraph_lines = DIGRAPH_LINE.findall(view_nofence) if DIGRAPH_ANY.search(view_nofence) else []
    digraph_chars = sum(len(l) for l in digraph_lines)
    symbol_chars = len(SYMBOLS.findall(QUOTE_PREFIX.sub('', view) if '>' in view else view))
    json_pairs = len(JSON_PAIR.findall(view_nofence)) if '":' in view_nofence or '" :' in view_nofence else 0
    bibtex_fields = len(BIBTEX_FIELD.findall(view_nofence)) if ' = ' in view_nofence or '={' in view_nofence or '="' in view_nofence else 0
    cov.update(code_lines=len(code_lines), code_chars=code_chars, code_cov=_ratio(code_chars, vn),
               fence_blocks=fences, fence_chars=fence_chars, fence_cov=_ratio(fence_chars, n),
               digraph_lines=len(digraph_lines), digraph_cov=_ratio(digraph_chars, vn),
               symbol_mass=_ratio(symbol_chars, vn),
               json_pairs=json_pairs, json_pairs_per_1k=json_pairs * 1000.0 / vn, bibtex_fields=bibtex_fields)
    flags['code_line_shape'] = ramp(len(code_lines), *p['code_lines']) * ramp(cov['code_cov'], *p['code_cov'])
    flags['code_fenced'] = ramp(cov['fence_cov'], *p['fence_cov'])
    flags['code_digraphs'] = ramp(len(digraph_lines), *p['digraph_lines']) * ramp(cov['digraph_cov'], *p['digraph_cov'])
    flags['code_symbol_mass'] = ramp(cov['symbol_mass'], *p['symbol_mass'])
    flags['code_json_pairs'] = ramp(json_pairs, *p['json_pairs']) * ramp(cov['json_pairs_per_1k'], *p['json_pairs_per_1k'])
    flags['code_bibtex'] = ramp(bibtex_fields, *p['bibtex_fields'])
    code = family([flags['code_line_shape'], flags['code_fenced'], flags['code_digraphs'], flags['code_symbol_mass'],
                   flags['code_json_pairs'], flags['code_bibtex']])

    # ---------------- math family ----------------
    latex_cmds = len(LATEX_CMD.findall(view)) if '\\' in view else 0
    latex_chars = sum(m.end() - m.start() for m in LATEX_SPAN.finditer(view)) if latex_cmds else 0
    dollar_spans = 0
    dollar_chars = 0
    if '$' in view:
        for m in DOLLAR_SPAN.finditer(view):
            span = m.group(1)
            if CURRENCY_START.match(span) and not MATH_MARK.search(span):
                continue          # "$12 and the meal cost $" is currency, not math
            if span.count(' ') > 12:
                continue          # a long prose run between two dollar signs
            bare = span.strip()
            if MATHLIKE.search(span) or (len(bare) <= 2 and bare.isalpha()):   # $x$, $n$
                dollar_spans += 1
                dollar_chars += m.end() - m.start()
    # One non-ASCII pass feeds: unicode math, isolated Greek, non-ASCII letters, mojibake, U+FFFD.
    non_ascii = NON_ASCII.findall(text)
    unicode_math = 0
    greek = 0
    non_ascii_letters = 0
    mojibake = 0
    replacement = 0
    foreign_letters = 0
    fancy_alnum = 0
    if non_ascii:
        joined = ''.join(non_ascii)
        unicode_math = len(UNICODE_MATH.findall(joined))
        foreign_letters = len(FOREIGN_SCRIPT.findall(joined))
        fancy_alnum = len(FANCY_ALNUM.findall(joined))
        non_ascii_letters = sum(1 for c in non_ascii if c.isalpha())
        replacement = joined.count('\ufffd')
        if GREEK_ANY.search(joined):
            greek = len(GREEK_ISOLATED.findall(view))
        if MOJIBAKE_ANY.search(joined):
            mojibake = len(MOJIBAKE.findall(text))
    unicode_math += greek
    eq_lines = EQUATION_LINE.findall(view) if EQ_MARK.search(view) else []
    inline_eq = 0
    if '=' in view:
        for m in INLINE_ASSIGN.finditer(view):
            i = m.start() - 1
            while i >= 0 and view[i] == ' ':
                i -= 1
            if i >= 0 and view[i].isalpha() and (i == 0 or not view[i - 1].isalnum()):
                inline_eq += 1
    if '^' in view:
        for m in CARET_POWER.finditer(view):
            i = m.start() - 1
            if i >= 0 and view[i].isalpha() and (i == 0 or not view[i - 1].isalnum()):
                inline_eq += 1
    if '(' in view:
        for m in FUNC_CALL.finditer(view):
            i = m.start() - 1
            if i >= 0 and view[i].isalpha() and (i == 0 or not view[i - 1].isalnum()):
                inline_eq += 1
    cov.update(latex_cmds=latex_cmds, latex_cov=_ratio(latex_chars, vn),
               dollar_spans=dollar_spans, dollar_cov=_ratio(dollar_chars, vn),
               unicode_math=unicode_math, unicode_math_per_1k=unicode_math * 1000.0 / vn,
               equation_lines=len(eq_lines), inline_eq=inline_eq, inline_eq_per_1k=inline_eq * 1000.0 / vn)

    # ---------------- equation-soup family ----------------
    soup_windows = 0
    soup_tokens = 0
    mix_total = len(MIX_TOKEN.findall(view))
    if mix_total >= 4:
        for line in view.split('\n'):
            if len(MIX_TOKEN.findall(line)) < 2:
                continue
            toks = line.split()
            n_t = len(toks)
            if n_t < 6:
                continue
            kinds = []
            for t in toks:
                m = SOUP_TOKEN.match(t)
                kinds.append(m.lastgroup if m else None)
            marks = [False] * n_t
            starts = range(0, n_t - SOUP_WINDOW + 1, SOUP_STRIDE) if n_t > SOUP_WINDOW else (0,)
            for st in starts:
                seg = kinds[st:st + SOUP_WINDOW]
                k = len(seg)
                soupy = sum(1 for x in seg if x)
                if k >= 6 and soupy >= 0.4 * k and sum(1 for x in seg if x == 'mix') >= 2:
                    soup_windows += 1
                    for i in range(st, st + k):
                        marks[i] = True
            soup_tokens += sum(marks)
    eq_markers = 0
    if '(' in view:
        own_line = len(EQ_MARKER_LINE.findall(view))
        inline = len(EQ_MARKER.findall(view)) - own_line
        eq_markers = own_line + (inline if soup_windows >= 1 else 0)
    total_tokens = view.count(' ') + view.count('\n') + 1

    # ---------------- soft structural flags ----------------
    words = WORDS.findall(text)
    nw = len(words)
    letters = sum(len(w) for w in words) + non_ascii_letters
    digits = 0
    controls = replacement
    for run in DIGIT_OR_CONTROL.findall(text):
        if run[0].isdigit():
            digits += len(run)
        else:
            controls += len(run)
    lines = [l.strip() for l in text.split('\n')]
    nonblank_lines = [l for l in lines if l]
    nb = max(1, len(nonblank_lines))
    cov['equation_share'] = _ratio(len(eq_lines), nb)
    long_lines = [l for l in nonblank_lines if len(l) >= 30]
    dup_chars = 0
    if long_lines:
        c = Counter(long_lines)
        dup_chars = sum(len(l) * (k - 1) for l, k in c.items() if k > 1)
    short_lines = sum(1 for l in nonblank_lines if l.count(' ') < 3) if len(nonblank_lines) >= 8 else 0
    top_word = top_tri = 0.0
    if nw:
        lower = [w.lower() for w in words[:NGRAM_WINDOW]]
        nl = len(lower)
        c_word = Counter(lower).most_common(1)[0][1]
        top_word = c_word / nl if c_word >= 5 else 0.0         # a share needs repetition, not a short text
        if nl >= 3:
            c_tri = Counter(zip(lower, lower[1:], lower[2:])).most_common(1)[0][1]
            top_tri = c_tri * 3 / nl if c_tri >= 3 else 0.0
    specials = len(SPECIAL_TOKEN.findall(text)) if '<|' in text else 0
    cov.update(soup_windows=soup_windows, soup_share=_ratio(soup_tokens, max(1, total_tokens)),
               mix_tokens=mix_total, eq_markers=eq_markers, eq_markers_per_1k=eq_markers * 1000.0 / vn)
    cov.update(words=nw, letters=letters, letter_density=_ratio(letters, n),
               non_ascii_letter_share=_ratio(non_ascii_letters, max(1, letters)),
               digit_share=_ratio(digits, max(1, digits + letters)),
               lines=len(nonblank_lines), repeated_line_mass=_ratio(dup_chars, n),
               short_line_share=_ratio(short_lines, nb),
               top_word_share=top_word, top_trigram_share=top_tri,
               control_chars=controls, special_tokens=specials, mojibake=mojibake,
               foreign_letters=foreign_letters, fancy_alnum=fancy_alnum)

    flags['math_latex'] = ramp(latex_cmds, *p['latex_cmds']) * ramp(cov['latex_cov'], *p['latex_cov'])
    flags['math_dollar'] = ramp(dollar_spans, *p['dollar_spans']) * ramp(cov['dollar_cov'], *p['dollar_cov'])
    flags['math_unicode'] = ramp(unicode_math, *p['unicode_math_count']) * ramp(cov['unicode_math_per_1k'], *p['unicode_math_per_1k'])
    flags['math_equation_lines'] = ramp(len(eq_lines), *p['equation_lines']) * ramp(cov['equation_share'], *p['equation_share'])
    flags['math_inline'] = ramp(inline_eq, *p['inline_eq_count']) * ramp(cov['inline_eq_per_1k'], *p['inline_eq_per_1k'])
    math = family([flags['math_latex'], flags['math_dollar'], flags['math_unicode'],
                   flags['math_equation_lines'], flags['math_inline']])
    flags['soup_windows'] = ramp(soup_windows, *p['soup_windows']) * ramp(cov['soup_share'], *p['soup_share'])
    flags['soup_eq_markers'] = ramp(eq_markers, *p['eq_markers']) * ramp(cov['eq_markers_per_1k'], *p['eq_markers_per_1k'])
    soup = family([flags['soup_windows'], flags['soup_eq_markers']])
    flags['script_foreign'] = ramp(foreign_letters, *p['foreign_letters'])
    flags['script_fancy'] = ramp(fancy_alnum, *p['fancy_alnum'])
    script = family([flags['script_foreign'], flags['script_fancy']])

    def soft(name, x):
        lo, hi, w = p[name]
        s = ramp(x, lo, hi)
        flags[name] = s
        return w * s

    soft_sum = 0.0
    soft_sum += soft('short_doc', 1.0 - min(1.0, nw / max(1, min_words)))
    soft_sum += soft('letter_density', 0.60 - cov['letter_density'])
    soft_sum += soft('non_ascii_letters', cov['non_ascii_letter_share'])
    soft_sum += soft('digit_share', cov['digit_share'])
    soft_sum += soft('repeated_lines', cov['repeated_line_mass'])
    soft_sum += soft('top_trigram', top_tri)
    soft_sum += soft('top_word', top_word)
    soft_sum += soft('short_lines', cov['short_line_share'])
    soft_sum += soft('url_count', len(urls))
    soft_sum += soft('url_mass', cov['url_cov'] * ramp(len(urls), 1, 3))   # one URL never counts by mass
    soft_sum += soft('email_count', emails)
    soft_sum += soft('control_chars', controls)
    soft_sum += soft('special_tokens', specials)
    soft_sum += soft('mojibake_per_1k', (mojibake * 1000.0 / n) * ramp(mojibake, 1, 3))
    soft_total = min(p['soft_cap'], soft_sum)

    axes = {'code': code, 'math': math, 'soup': soup, 'script': script, 'html': html, 'soft': soft_total}
    bad = combine(axes)
    return {'bad_score': bad, 'flags': flags, 'coverage': cov, 'axes': axes,
            'decision': decide(bad, p['thresholds']), 'filter_version': VERSION}


def combine(axes):
    """bad = min(1, max(code, math, soup, script, html) + soft). Monotone in every axis."""
    hard = max(axes['code'], axes['math'], axes.get('soup', 0.0), axes.get('script', 0.0), axes['html'])
    return round(min(1.0, hard + axes['soft']), 6)


def decide(bad, thresholds=None):
    t = PARAMS['thresholds'] if thresholds is None else thresholds
    if bad < t['accept']:
        return 'accept'
    if bad < t['reject']:
        return 'borderline'
    return 'reject'


if __name__ == '__main__':
    import json
    import sys
    for line in sys.stdin:
        r = classify_v4(json.loads(line)['text'] if line.startswith('{') else line.rstrip('\n'))
        print(json.dumps({'bad_score': r['bad_score'], 'decision': r['decision'],
                          'axes': r['axes']}))
