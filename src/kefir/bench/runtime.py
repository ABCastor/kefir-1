"""Run the frozen suite against a backend; scoring stays separate (``scoring``)."""
import time


def evaluate_cases(cases, backend):
    """Admit all inputs before inference; never silently truncate or drop a case."""
    for c in cases:
        backend.admit(c)
    rows=[]
    start=time.monotonic()
    for c in cases:
        if c['category']=='grammar':
            row={'id':c['id'],'logp_good':backend.score_raw(c['sentence_good']),
                 'logp_bad':backend.score_raw(c['sentence_bad'])}
        else:
            row={'id':c['id'],**backend.generate(c)}
            row.update(sum_logprob=backend.score_answer(c,c['gold'][0]),
                       gold_bytes=len(c['gold'][0].encode('utf-8')))
            if 'choices' in c:
                row['choice_logps']=[backend.score_cloze(c,choice) for choice in c['choices']]
                row['choice_char_lengths']=[len(x) for x in c['choices']]
        rows.append(row)
    return rows,{'backend':backend.identity,'wall_seconds':time.monotonic()-start,
                 'generated_tokens':sum(r.get('generated_tokens',0) for r in rows),
                 'input_tokens':sum(r.get('input_tokens',0) for r in rows),
                 'admission':'all inputs plus output allowance fit; no truncation'}
