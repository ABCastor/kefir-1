"""Read, validate and fingerprint benchmark artifacts without model imports."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

CATEGORIES = {'knowledge', 'grammar', 'reasoning', 'instructions', 'context', 'stopping'}
SPLITS = {'dev', 'test', 'reserve'}

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_jsonl(path):
    def strict_constant(value):
        raise ValueError(f'nonfinite JSON constant {value}')
    rows=[]
    for number,line in enumerate(Path(path).read_text().splitlines(),1):
        if not line.strip():
            raise ValueError(f'{path}:{number}: blank JSONL record')
        row=json.loads(line, parse_constant=strict_constant)
        if not isinstance(row,dict):
            raise ValueError(f'{path}:{number}: object required')
        rows.append(row)
    return rows

def write_jsonl(path, rows):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')

def validate_cases(cases):
    seen=set(); families=defaultdict(set); pairs=defaultdict(list); counts=Counter()
    for c in cases:
        for field in ['id','split','family_id','category','source','max_new_tokens','gold','content_mode','constraints']:
            if field not in c: raise ValueError(f'missing case field {field}')
        if not isinstance(c['id'],str) or not c['id'] or c['id'] in seen: raise ValueError('empty or duplicate case ID')
        seen.add(c['id'])
        if c['split'] not in SPLITS or c['category'] not in CATEGORIES: raise ValueError('invalid split/category')
        if not isinstance(c['family_id'],str) or not c['family_id']: raise ValueError('family_id required')
        if not isinstance(c['source'],dict): raise ValueError('source metadata required')
        if not isinstance(c['gold'],list) or not c['gold'] or any(not isinstance(s,str) or not s.strip() for s in c['gold']): raise ValueError('nonempty gold answers required')
        if not isinstance(c['constraints'],list): raise ValueError('constraint list required')
        if c['content_mode'] not in {'exact','review','grammar'}: raise ValueError('invalid content_mode')
        counts[(c['split'],c['category'])]+=1
        if c['source'].get('kind')=='authored' or c['source'].get('dataset')=='authored':
            families[c['family_id']].add(c['split'])
        if c['category']=='grammar':
            if c['content_mode']!='grammar' or c['max_new_tokens']!=0: raise ValueError('grammar scoring contract')
            if not all(isinstance(c.get(k),str) and c[k].strip() for k in ['sentence_good','sentence_bad']): raise ValueError('grammar pair required')
        else:
            if type(c['max_new_tokens']) is not int or c['max_new_tokens']<=0: raise ValueError('positive output cap required')
            messages=c.get('messages')
            if not isinstance(messages,list) or not messages or messages[-1].get('role')!='user': raise ValueError('prompt must end in user')
            expected='user'
            for i,m in enumerate(messages):
                if not isinstance(m,dict) or not isinstance(m.get('content'),str) or not m['content'].strip(): raise ValueError('nonempty message required')
                role=m.get('role')
                if role=='system' and i==0: continue
                if role!=expected: raise ValueError('alternating messages required')
                expected='assistant' if role=='user' else 'user'
        if 'pair_id' in c: pairs[(c['split'],c['pair_id'])].append(c)
    if any(len(v)>1 for v in families.values()): raise ValueError('authored family crosses splits')
    for key,items in pairs.items():
        if len(items)!=2 or len({x.get('variant') for x in items})!=2 or len({x['gold'][0].casefold() for x in items})!=2:
            raise ValueError(f'context pair must have two distinct variants and answers: {key}')
    return {'cases':len(cases),'by_split_category':{f'{s}/{c}':n for (s,c),n in sorted(counts.items())},'context_pairs':len(pairs)}
