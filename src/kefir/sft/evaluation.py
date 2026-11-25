"""SFT evaluation: held-out Assistant NLL, the frozen chat suite and raw retention.

The backend is the in-memory model adapter the Kefir 1 SFT runs used at every
milestone, with one change of authority: prompts come from the one Renderer C
(``kefir.sft.renderer``) and text from the pinned Kefir tokenizer, instead of the two
private copies the run-time adapter carried. Their equality on the whole frozen suite
is recorded in ``docs/code-record.md``.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from kefir.tokenizer import TOKENIZER_SHA256, load_tokenizer

from .renderer import EOD, MSG_END, ChatRenderer


def check_precision(precision: str, device: str) -> None:
    """The one rule for SFT evaluation precision: fp32 anywhere, fp16 on CUDA only."""
    if precision not in {'fp16', 'fp32'}:
        raise ValueError('unqualified evaluation precision')
    if precision == 'fp16' and not str(device).startswith('cuda'):
        raise ValueError(f'fp16 evaluation requires a CUDA device, not {device}; use fp32')


class ModelBackend:
    """Greedy generation and likelihoods over a loaded Kefir model (no trainer state)."""

    def __init__(self, model, *, device=None, precision="fp32", context_limit=2048):
        import torch
        self.torch = torch
        self.model = model
        self.device = str(device if device is not None else next(model.parameters()).device)
        self.renderer = ChatRenderer(load_tokenizer())
        self.limit = context_limit
        if not 1 <= self.limit <= 2048:
            raise ValueError('invalid context limit')
        check_precision(precision, self.device)
        self.precision = precision
        self.identity = {'kind': 'in_memory_k1', 'tokenizer_sha256': TOKENIZER_SHA256,
                         'renderer': 'kefir-chat-c-v1',
                         'renderer_sha256': self.renderer.identity['renderer_sha256'],
                         'context_limit': self.limit, 'precision': self.precision,
                         'decoding': 'greedy; native EOD; full forward; output IDs 0..7992',
                         'teacher_forcing': 'gold continuation encoded separately from prompt'}

    def encode(self, text):
        return self.renderer.encode_content(text)

    def prompt(self, case):
        return self.renderer.generation_prompt(case['messages'])

    def cloze_parts(self, case, text):
        q = case.get('source_question', case.get('original_question',
                                                 case.get('question', case['messages'][-1]['content'])))
        return [EOD] + self.encode('Question: ' + q + '\nAnswer:'), self.encode(' ' + text)

    def admit(self, case):
        if case['category'] == 'grammar':
            if max(1 + len(self.encode(case[k])) for k in ('sentence_good', 'sentence_bad')) > self.limit:
                raise ValueError('grammar overflow: ' + case['id'])
            return
        prefix = self.prompt(case)
        if len(prefix) + max(case['max_new_tokens'], len(self.encode(case['gold'][0]))) > self.limit:
            raise ValueError('benchmark overflow: ' + case['id'])
        for choice in case.get('choices', []):
            p, t = self.cloze_parts(case, choice)
            if len(p) + len(t) > self.limit:
                raise ValueError('choice overflow')

    def logits(self, ids, *, full_vocab=False):
        torch = self.torch
        with torch.inference_mode(), torch.autocast(
                device_type='cuda', dtype=torch.float16, enabled=self.precision == 'fp16'):
            logits = self.model(torch.tensor([ids], device=self.device, dtype=torch.long))
        if logits.ndim != 3 or logits.shape[-1] < EOD + 1:
            raise ValueError('unexpected logits')
        return (logits[0] if full_vocab else logits[0, :, :EOD + 1]).float()

    def _score(self, prefix, target):
        if not prefix or not target or len(prefix) + len(target) > self.limit:
            raise ValueError('invalid scoring length')
        logits = self.logits(prefix + target)[len(prefix) - 1:-1]
        targets = self.torch.tensor(target, device=self.device, dtype=self.torch.long)
        value = float(logits.log_softmax(-1).gather(1, targets[:, None]).sum().item())
        if not math.isfinite(value):
            raise ValueError('nonfinite log likelihood')
        return value

    def score_raw(self, text):
        return self._score([EOD], self.encode(text))

    def score_answer(self, case, text):
        return self._score(self.prompt(case), self.encode(text))

    def score_cloze(self, case, text):
        return self._score(*self.cloze_parts(case, text))

    def generate(self, case):
        prefix = self.prompt(case)
        tokens = []
        for _ in range(case['max_new_tokens']):
            tokens.append(int(self.logits(prefix + tokens)[-1].argmax()))
            if tokens[-1] == EOD:
                break
        ended = bool(tokens and tokens[-1] == EOD)
        return {'text': self.renderer.tokenizer.decode(tokens[:-1] if ended else tokens),
                'termination': 'native_end' if ended else 'output_cap', 'token_ids': tokens,
                'input_tokens': len(prefix), 'generated_tokens': len(tokens)}


def validated_row(row, limit):
    ids, mask = row.get('input_ids'), row.get('assistant_mask')
    if not isinstance(ids, list) or not isinstance(mask, list) or len(ids) != len(mask) or not 2 <= len(ids) <= limit:
        raise ValueError('invalid heldout tokenized row')
    if any(type(x) is not int or not 0 <= x <= MSG_END for x in ids) or any(type(x) is not bool for x in mask):
        raise ValueError('invalid heldout IDs/mask')
    if mask[0] or not any(mask[1:]) or any(m and token > EOD for token, m in zip(ids, mask)):
        raise ValueError('invalid heldout assistant targets')
    return ids, mask


def validation_loss(backend, rows):
    """Mean NLL over every held-out Assistant target (text and EOD), full vocabulary."""
    total = 0.
    count = 0
    chats = 0
    torch = backend.torch
    for row in rows:
        ids, mask = validated_row(row, backend.limit)
        logits = backend.logits(ids[:-1], full_vocab=True)
        target = torch.tensor(ids[1:], device=backend.device, dtype=torch.long)
        selected = torch.tensor(mask[1:], device=backend.device, dtype=torch.bool)
        nll = -logits.log_softmax(-1).gather(1, target.clamp_max(EOD)[:, None]).squeeze(1)[selected].sum()
        total += float(nll.item())
        count += sum(mask[1:])
        chats += 1
    if not count or not math.isfinite(total):
        raise ValueError('empty/nonfinite heldout loss')
    return {'assistant_nll_sum': total, 'assistant_target_count': count, 'assistant_nll': total / count,
            'perplexity': math.exp(total / count) if total / count < 700 else None, 'conversations': chats,
            'denominator': 'all heldout assistant text and EOD target tokens; no router auxiliary loss'}


def _raw_case(case):
    if case.get('split') != 'dev' or case.get('category') != 'knowledge':
        raise ValueError('only admitted development knowledge cases')
    source = case.get('source', {}).get('dataset')
    if source == 'ARC-Easy':
        question = case['source_question']
    elif source == 'PopQA':
        messages = case['messages']
        suffix = '\nAnswer briefly.'
        if len(messages) != 1 or messages[0]['role'] != 'user' or not messages[0]['content'].endswith(suffix):
            raise ValueError('PopQA raw question requires the frozen one-turn template')
        question = messages[0]['content'][:-len(suffix)]
    else:
        raise ValueError('unreviewed source')
    if not isinstance(question, str) or not question.strip():
        raise ValueError('empty raw question')
    return {**case, 'source_question': question}


def raw_retention(cases, backend):
    """Gold and choice likelihood under the raw ``Question: ...\\nAnswer:`` view.

    Reports gold bits per UTF-8 byte; likelihood is not binary factual correctness.
    """
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('empty or duplicate cases')
    out = []
    for original in cases:
        case = _raw_case(original)
        gold = case['gold'][0]
        prefix, target = backend.cloze_parts(case, gold)
        expected = [EOD] + backend.encode('Question: ' + case['source_question'] + '\nAnswer:')
        if prefix != expected or target != backend.encode(' ' + gold):
            raise ValueError('raw prefix/continuation contract mismatch')
        if not target or len(prefix) + len(target) > backend.limit:
            raise ValueError('raw overflow or empty target')
        choices = case.get('choices')
        if choices:
            if case['gold_choice'] not in range(len(choices)) or gold != choices[case['gold_choice']]:
                raise ValueError('gold and candidate text differ')
            for choice in choices:
                p, t = backend.cloze_parts(case, choice)
                if p != prefix or not t or len(p) + len(t) > backend.limit:
                    raise ValueError('candidate overflow or changed prefix')
            values = [backend.score_cloze(case, text) for text in choices]
            gold_logp = values[case['gold_choice']]
        else:
            values = None
            gold_logp = backend.score_cloze(case, gold)
        if not math.isfinite(gold_logp) or gold_logp > 0 or (values and any(not math.isfinite(x) or x > 0 for x in values)):
            raise ValueError('invalid raw log probability')
        row = {'id': case['id'],
               'case_sha256': hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest(),
               'prefix_ids': prefix, 'target_ids': target, 'canonical_gold': gold,
               'gold_logprob': gold_logp, 'target_utf8_bytes': len((' ' + gold).encode('utf-8')),
               'choice_logps': values, 'choice_char_lengths': [len(x) for x in choices] if choices else None}
        if values:
            g = case['gold_choice']
            for name, v in [('raw', values), ('character_normalized', [x / len(t) for x, t in zip(values, choices)])]:
                winners = [i for i, x in enumerate(v) if x == max(v)]
                row[name] = {'gold_minus_best_distractor': v[g] - max(x for i, x in enumerate(v) if i != g),
                             'credit': 1 / len(winners) if g in winners else 0, 'tie_count': len(winners)}
        out.append(row)
    return {'protocol': 'raw-retention-dev-v1', 'backend': backend.identity, 'cases': len(out), 'rows': out,
            'gold_bpb': -sum(r['gold_logprob'] for r in out) / (math.log(2) * sum(r['target_utf8_bytes'] for r in out)),
            'denominator': 'UTF-8 bytes of separately encoded leading space plus canonical answer; no EOD target',
            'limitations': ['Likelihood is not binary factual correctness.', 'Only development evidence.',
                            'Same token support/precision and attributable parent-child required for a paired claim.']}


def evaluate_model(model, *, cases, validation_rows=None, precision="fp32", device=None) -> dict:
    """The one SFT evaluation: suite responses and scores, raw retention, optional NLL."""
    from kefir.bench.runtime import evaluate_cases
    from kefir.bench.scoring import aggregate, score_response
    if model.training:
        raise ValueError('caller must set evaluation mode')
    backend = ModelBackend(model, device=device, precision=precision)
    responses, receipt = evaluate_cases(cases, backend)
    scored = [score_response(c, r) for c, r in zip(cases, responses)]
    knowledge = [c for c in cases if c['category'] == 'knowledge' and c.get('split') == 'dev']
    retention = raw_retention(knowledge, backend) if knowledge else None
    return {
        'summary': {'cases': len(cases), 'benchmark': aggregate(cases, scored),
                    'raw_retention_gold_bpb': retention['gold_bpb'] if retention else None,
                    'validation': validation_loss(backend, validation_rows) if validation_rows else None,
                    'runtime': receipt,
                    'limitations': ['Exact open-answer knowledge scores are lower bounds, not semantic accuracy.',
                                    'Template mechanisms are partly shared between authored splits.',
                                    'Public benchmark pretraining contamination is unknown.',
                                    'No overall score, release thresholds or automatic winner.']},
        'responses': responses, 'scored': scored, 'retention': retention,
    }


__all__ = ["ModelBackend", "evaluate_model", "raw_retention", "validated_row", "validation_loss"]
