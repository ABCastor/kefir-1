"""English-prose selection v1. Heuristics, not a certificate of semantic purity.

Reject whole documents: removing a formula can silently corrupt the explanation.
No source corpus or trainer is modified. Byte coverage is a tokenizer property,
not a reason to retain code or to delete ordinary English punctuation.
"""
import re
import unicodedata
from collections import Counter

VERSION = 'english-prose-v1'
WORDS = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
LATEX = re.compile(r'\\[A-Za-z]+\b|\\[\[\](){}]|\$\$|\$[^\n$]{1,300}\$|\\(?:\^|_)')
SYMBOLS = re.compile(r'[∀-⋿⟀-⣿𝔀-𝟿]|[α-ωΑ-Ω]|\^\s*\{?[-+\dA-Za-z]|\b\w+_\{')
CODE = re.compile(r'''(?im)```|~~~|^\s*(?:def\s+\w+\s*\(|class\s+\w+[^\n]*:|(?:from\s+[\w.]+\s+)?import\s+[\w.]+|#include\s*[<"]|(?:public|private|protected)\s+(?:static\s+)?(?:void|class|int)\b|function\s*\w*\s*\(|(?:const|let|var)\s+\w+\s*=|(?:SELECT\s+.+\s+FROM|INSERT\s+INTO|CREATE\s+TABLE)\b|(?:sudo|pip|npm|apt-get)\s+\w+)|\b(?:console\.log|print|range|len|eval|exec|np\.\w+|pd\.\w+|plt\.\w+)\s*\(|=>|:=|==|!=|&&|\|\||\w+\[[\w'"\d]+\]\s*=|</?[A-Za-z][^>]{0,150}>''')
EQUATION = re.compile(r'(?:\b[A-Za-z0-9]+\s*[-+*/]\s*[A-Za-z0-9]+\s*=|\b[A-Za-z0-9]+\s*=\s*[-+]?\s*[A-Za-z0-9(]|\b\d+\s*[+*/]\s*\d+|\b[A-Za-z]\s*[<>]\s*[A-Za-z\d]|\b(?:sin|cos|log|exp)\s*\(|[{}]|\b[A-Za-z]\s*\^\s*\d)')
PROGRAMMING = re.compile(r'\b(?:javascript|typescript|C\+\+|programming|source code|computer code|compiler|debugging|code snippet|SQL query|Python (?:code|script|function|program)|software development|formal logic|predicate calculus|propositional calculus|lambda calculus)\b', re.I)
PII = re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b|\b\d{3}-\d{2}-\d{4}\b|\b(?:api[_ -]?key|password|secret[_ -]?key)\s*[:=]\s*\S+', re.I)
URL = re.compile(r'https?://\S+|www\.\S+', re.I)
SPECIAL = re.compile(r'<\|[^>]+\|>|</?(?:think|answer)>')
BUCKETS = {
 'physical_world': r'\b(?:gravity|friction|falls?|falling|floats?|floating|boils?|boiling|melts?|melting|solid|liquid|temperature|heat|motion)\b',
 'spatial_reasoning': r'\b(?:inside|outside|underneath|above|below|beside|behind|direction|distance|left|right|position)\b',
 'verbal_reasoning': r'\b(?:if|unless|therefore|because|consequently|suppose|imagine|conclude|riddle|puzzle)\b',
 'geography': r'\b(?:capital|country|countries|continent|geography|island|ocean|border|province|city|cities)\b',
 'language_grammar': r'\b(?:grammar|grammatical|pronoun|noun|verb|adjective|adverb|sentence|punctuation|pronunciation|vocabulary)\b',
 'biology_health': r'\b(?:biology|cells?|organism|animals?|plants?|health|disease|nutrition|body|brain|medicine)\b',
 'history_culture': r'\b(?:history|historical|ancient|century|empire|tradition|culture|civilization|archaeology)\b',
 'society_economics': r'\b(?:economy|economic|government|society|community|citizens|trade|employment|democracy|income)\b',
 'arts_literature': r'\b(?:literature|novel|poetry|poem|painting|music|theatre|theater|artist|story|character)\b',
 'practical_life': r'\b(?:cooking|recipe|garden|gardening|cleaning|repair|kitchen|furniture|travel|shopping)\b',
 'conversation_emotions': r'\b(?:feel|feeling|happy|sad|friend|friendship|conversation|hello|thank|please|sorry|relationship)\b',
}
TOPICS = {k: re.compile(v, re.I) for k,v in BUCKETS.items()}

def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())

def topics(text):
    counts = {k: len(p.findall(text)) for k,p in TOPICS.items()}
    found = [k for k,v in counts.items() if v >= 3]
    return sorted(found, key=lambda k: (-counts[k], k)) or ['general_prose']

def syntax_reason(text):
    for name, pattern in [('special_token_literal', SPECIAL), ('latex', LATEX), ('symbolic_math', SYMBOLS),
                          ('code_or_markup', CODE), ('equation', EQUATION), ('programming_domain', PROGRAMMING), ('pii_pattern', PII)]:
        if pattern.search(text): return name
    return None

class ProseFilter:
    def __init__(self, language_model):
        self.language_model = language_model

    def assess(self, text):
        # Numeric thresholds are conservative engineering policy, not measured optimum.
        if not isinstance(text, str): return 'not_text', {}
        if len(text) > 100000: return 'oversize_document', {}
        reason = syntax_reason(text)
        if reason: return reason, {}
        if '\ufffd' in text or any(ord(c)<32 and c not in '\n\t\r' for c in text): return 'invalid_control', {}
        words = WORDS.findall(text)
        if len(words) < 40: return 'short_fragment', {}
        letters = sum(c.isalpha() for c in text)
        if letters / max(1,len(text)) < .60: return 'low_prose_density', {}
        if sum(c.isalpha() and not c.isascii() for c in text)/max(1,letters) > .02: return 'non_latin_density', {}
        if len(URL.findall(text)) > 2: return 'link_heavy', {}
        counts = Counter(w.lower() for w in words)
        if counts.most_common(1)[0][1]/len(words) > .18: return 'word_repetition', {}
        paragraphs = [normalized(p) for p in text.split('\n') if len(p.strip()) > 40]
        if paragraphs and 1-len(set(paragraphs))/len(paragraphs) > .20: return 'paragraph_repetition', {}
        trigrams = Counter(zip(words, words[1:], words[2:]))
        if trigrams and trigrams.most_common(1)[0][1]*3/len(words) > .15: return 'ngram_repetition', {}
        # Scan ALL chunks, so a long English prefix cannot hide a foreign appendix.
        # Minimum40 words applies to the document, not to headings/proper names.
        chunks = [text[i:i+2000].replace('\n',' ') for i in range(0,len(text),2000)]
        scores = []
        for chunk in chunks:
            if len(WORDS.findall(chunk)) < 20: continue
            labels, probs = self.language_model.predict(chunk, k=1)
            if labels[0] != '__label__en' or float(probs[0]) < .90: return 'language_confidence', {}
            scores.append(float(probs[0]))
        if not scores: return 'language_unverified', {}
        return None, {'english_min_score': min(scores), 'words':len(words), 'topics':topics(text)}
