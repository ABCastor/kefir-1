"""Load the frozen public subset without network access.

``build_public(output_dir, cache_dir=None)`` verifies and writes the selected
JSONL files and source manifest, then returns their cases. ``cache_dir`` is
reserved for callers using a common build interface; it is not accessed.
The bundled selected data are the build inputs, not an upstream download.
The manifest pins the original files and records selection rules and limitations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data"
_MANIFEST = "source-manifest.json"


def build_public(output_dir, cache_dir=None):
    """Verify and materialize the frozen subset; return its case dictionaries.

    Exact answer matching is a lower bound. ARC open answers need independent
    semantic adjudication, and the original choice-ranking view is separate.
    """
    del cache_dir
    manifest_bytes = (_DATA / _MANIFEST).read_bytes()
    manifest = json.loads(manifest_bytes)
    payloads = {}
    cases = []
    for name, expected in manifest["selected_files"].items():
        if Path(name).name != name:
            raise ValueError("invalid bundled public data filename")
        payload = (_DATA / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"public data hash mismatch: {name}")
        payloads[name] = payload
        cases.extend(json.loads(line) for line in payload.splitlines() if line.strip())
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate public case ID")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (output / name).write_bytes(payload)
    (output / _MANIFEST).write_bytes(manifest_bytes)
    return cases


def select_from_sources(cache_dir, output_dir):
    """Reproduce selection from the manifest-pinned upstream files.

    Download each manifest source URL into cache_dir under its file name and
    install pyarrow separately to read the two upstream Parquet files. This
    optional reconstruction path is distinct from the standard-library build.
    """
    import csv,json,hashlib,re,collections,sys
    from pathlib import Path
    import pyarrow.parquet as pq
    P=Path(cache_dir); D=Path(output_dir); D.mkdir(parents=True,exist_ok=True)
    original = json.loads((_DATA / _MANIFEST).read_bytes())
    for entry in original['sources']:
     data = (P / entry['file']).read_bytes()
     if hashlib.sha256(data).hexdigest() != entry['sha256']:
      raise ValueError('upstream hash mismatch: ' + entry['file'])
    def rank(s): return hashlib.sha256(('kefir-public-v1:'+str(s)).encode()).hexdigest()
    def jdump(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
    def mk(id,split,src,q,gold,**kw):return dict(id=id,split=split,family_id=id,category='knowledge',source=src,max_new_tokens=32,messages=[{'role':'user','content':q+'\nAnswer briefly.'}],gold=gold,content_mode='exact',constraints=[],**kw)
    with (P/'popqa.tsv').open() as f: rows=list(csv.DictReader(f,delimiter='\t'))
    pop_rows=rows
    allowed={'capital'}
    # Explicit original IDs: present-day countries and unambiguous state/province
    # capitals only. No birthplace rows: the source does not reliably establish
    # whether the referenced object is a city or uniquely identifies a person.
    geography_ids=set('2634781 1678688 6033282 2763878 5152223 523639 6259248 2139200 5904820 6432170 6117457 756915 19472 6216050 217415 6338836 5978636 3030578 2998350 1882504 6345816 1511830 43108 1763729 2611738 51112 2203391 5282300 2839162 552289 5795369 6133150 6256826 2950925 528873 2266274 504663 5592430 6049645 1802332 6266561 3108719 2986069 246233 6434714 6386307 52919 20250 894483 6368219 5895914 5806895 58303 6258460 6083055 6519044 1770493 1840196 4976747 1740588 1727462 1140403 2449609 35 1509169 307727 793141 2468664 1515801 600223 1867834 1734901 39178 28977 1580390 6291315 357926 546090 2939807 6270851 2972379 1685298 5844314 6549760 5210420 4995155 1826568 1449028 30821 745100 1665931 1660531 1022353 2254982 1860029 25674 1592930 861082 598694 6298839 2776008 2070124 534816 6313411 2583275 4394181 6257858 6145054 2967643 6026411 761251 715797 5175831 6409891 6499110 681897 701352 581753 2036135 1537651 1259497 1014859 6418167 1581906 6263500 1722391 6098623 1464242 56722 6300717 355719 6262109 2507186 5015775 6408743 1576431 6351301 6495030 562847 5547087 3008151 6330613 5864218 2725932 6359560 1717233 1363778 728749 6355626 1468996 1207361 1746524 203437 5776395 6273545 302854 15805 13046 37500 6526129 5925 1775298 5932260 6475333 1691938 1712488 557793 6083018 2712847 13998 239951 5815032 2658151 2438406 1388910 1672533 5421647 6505231 1752279 6279762 6322925 6604 44382 1706834 804606 5623381 6015165 1573571 593981 306840 297952 6402884 10956 5660051 6554019 832908 6305992 34246 2935548 542552 318560 654471 802449'.split())
    elig=[r for r in rows if r['id'] in geography_ids and r['prop'] in allowed and r['subj_id'] and r['s_pop'].isdigit() and r['obj'].strip() and len(r['obj'].split())<=5 and len(r['question'])<=180 and len(r['subj'])<=80 and r['question'].endswith('?')]
    # Popularity quartiles are computed on the eligible question pool before deduplication.
    vals=sorted(int(r['s_pop']) for r in elig); q50=vals[len(vals)//2];q75=vals[3*len(vals)//4]
    used=set(); selected=[]
    # round-robin relation types; sample subjects without replacement in each popularity stratum
    for split,n in [('dev',32),('test',64),('reserve',32)]:
     for quartile,num in [('upper_half',n*3//4),('lower_half',n//4)]:
      buckets={rel: sorted((r for r in elig if r['prop']==rel and ((int(r['s_pop'])>=q50) if quartile=='upper_half' else (int(r['s_pop'])<q50))),key=lambda r:rank(r['id'])) for rel in sorted(allowed)}
      for i in range(num):
       found=None
       for offset in range(len(buckets)):
        rel=sorted(buckets)[(i+offset)%len(buckets)]; b=buckets[rel]
        while b and b[0]['subj_id'] in used: b.pop(0)
        if b: found=b.pop(0);break
       if not found: raise ValueError('insufficient PopQA subject supply')
       used.add(found['subj_id']); r=found
       # Only the canonical source object is accepted. Source aliases are retained for review.
       src={'dataset':'PopQA','id':r['id'],'subject':r['subj'],'subject_id':r['subj_id'],'subject_wikipedia_title':r['s_wiki_title'],'relation':r['prop'],'subject_popularity':int(r['s_pop']),'popularity_stratum':quartile,'source_aliases':json.loads(r['possible_answers'])}
       title=r['s_wiki_title'].replace('_',' ')
       q=r['question']
       if title and title != r['subj'] and '(' in title and r['subj'] in q:
        q=q.replace(r['subj'],title,1)
       selected.append(mk('popqa-'+r['id'],split,src,q,[r['obj']],source_question=r['question'],adaptation_reference_status='unreviewed canonical source object; subject Wikipedia title inserted when it carries a disambiguating parenthesis; semantic alternatives require review'))
    arc=[]
    for source_split,alloc in [('validation',[('dev',30)]),('test',[('test',64),('reserve',32)])]:
     rows=pq.read_table(P/('arc_'+source_split+'.parquet')).to_pylist()
     def eligible(r):
      q=r['question']; c=r['choices']; key=r['answerKey']
      rejected_arc={'Mercury_7241343':'specialist molecular question','Mercury_7227763':'specialist molecular question','Mercury_7227850':'disputed virus protein-shell and cell-membrane analogy','Mercury_7040863':'excluded from compact elementary selection','Mercury_SC_405062':'wind is not always available','MCAS_2015_5_11':'ambiguous compass energy wording','Mercury_SC_402064':'gas production can change factors besides pressure'}
      return bool(r['id'] not in rejected_arc and (re.match(r'^(What|How|Why|Where|When|Name)\b',q) and q.endswith('?') and not re.search(r'following|below|above|diagram|picture|figure|chart|table|graph|these|best|most|example|shown|which of|this (experiment|investigation|procedure|test|study|model|system|object|organism|process|observation|result)|\b(genotype|phenotype|allele|homozygous|heterozygous|momentum|kilogram|formula)\b|\d|[A-Z][a-z]?\d+',q,re.I) and key in c['label'] and len(c['text'][c['label'].index(key)].split())<=6 and not re.search(r'\d|[A-Z][a-z]?\d+',c['text'][c['label'].index(key)]) and len(q)<=300 and len(c['text'])>=2))
     candidates=sorted((r for r in rows if eligible(r)),key=lambda r:rank(r['id'])); assert len(candidates)>=sum(n for _,n in alloc)
     index=0
     for split,n in alloc:
      for r in candidates[index:index+n]:
       c=r['choices']; k=c['label'].index(r['answerKey']); q=r['question']; answer=c['text'][k]
       arc.append(mk('arc-'+r['id'],split,{'dataset':'ARC-Easy','id':r['id'],'official_split':source_split},q,[answer],choices=c['text'],gold_choice=k,source_question=q,source_choice_labels=c['label'],adaptation_reference_status='unreviewed: source correct choice is a strict exact lower bound; other valid open answers require adjudication'))
      index+=n
    blimp=[]
    uids=['anaphor_gender_agreement','anaphor_number_agreement','determiner_noun_agreement_1','determiner_noun_agreement_2','irregular_past_participle_adjectives','irregular_past_participle_verbs','principle_A_c_command','npi_present_1']
    for uid in uids:
     rows=[json.loads(s) for s in (P/(uid+'.jsonl')).read_text().splitlines()]; rows.sort(key=lambda x:rank(uid+':'+str(x['pairID'])))
     for split,rr in [('dev',rows[:8]),('test',rows[8:24]),('reserve',rows[24:32])]:
      for r in rr:
       id='blimp-'+uid+'-'+str(r['pairID']); blimp.append(dict(id=id,split=split,family_id=id,category='grammar',source={'dataset':'BLiMP','uid':uid,'pair_id':r['pairID'],'linguistics_term':r['linguistics_term']},max_new_tokens=0,gold=[r['sentence_good']],content_mode='grammar',constraints=[],sentence_good=r['sentence_good'],sentence_bad=r['sentence_bad']))
    for name,records in [('public-popqa.jsonl',selected),('public-arc.jsonl',arc),('public-blimp.jsonl',blimp)]: (D/name).write_text(''.join(jdump(r)+'\n' for r in records))
    urls={entry['file']:entry['url'] for entry in original['sources']}; sources=[]
    for name,url in urls.items():
     data=(P/name).read_bytes(); sources.append({'file':name,'url':url,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
    man={'format_version':1,'sources':sources,'licenses':{'PopQA':'MIT (source repository LICENSE)','ARC-Easy':'CC BY-SA 4.0 (author dataset card)','BLiMP':'CC BY 4.0 (dataset card and author repository)'},'selection':{'seed':'sha256 kefir-public-v1','popqa':{'relations':sorted(allowed),'scope_note':'Narrowed from proposal to an explicit whitelist of present-day geographic entities and forward capitals only. Birthplaces omitted because entity and city validity cannot be established mechanically. Sampling relaxes the proposed quartile quota to upper and lower popularity halves to retain the planned counts. This is not broad general knowledge; references remain unreviewed.','eligible_count':len(elig),'median_popularity':q50,'upper_quartile_popularity':q75,'popularity_definition':'halves of eligible question rows before subject deduplication; cutoff ties go to upper half','subject_isolation':'subject IDs unique across all splits','aliases':'canonical object only; source answer aliases retained as unreviewed metadata','quotas':'75% upper half, 25% lower half per split; dev 32, test 64, reserve 32','eligible_source_ids':sorted(geography_ids),'rejected_source_ids':{r['id']:('historical, contested or date-sensitive subject' if r['prop']=='capital' else 'ambiguous identity, typo or non-city birth reference') for r in pop_rows if r['subj'] in {'Han dynasty','Tang Empire','Tokugawa Shogunate','Kingdom of Italy','Russian Empire','Czechoslovakia','German Democratic Republic','United Kingdom of Great Britain and Ireland','Song dynasty','South Africa','European Union','Indonesia','Burundi','Annie Oakley','Isaac','Isaac Newton','Roman Nose','Michelle Renaudddd','Bury','Nelson County'} and r['prop'] in {'capital','place of birth'}}},'arc':{'dev':'30 eligible items from official validation; fewer than proposed after exclusions','test_and_reserve':'official test, disjoint IDs','filter':'question starts What, How, Why, Where, When or Name; ends ?; excludes visual, choice-dependent and deictic markers, relative most/best, genetics notation, formulae, numbers and momentum terms as listed in selector; gold 6 words or fewer without digits','rejected_source_ids':{'Mercury_7241343':'specialist molecular question','Mercury_7227763':'specialist molecular question','Mercury_7227850':'disputed virus protein-shell and cell-membrane analogy','Mercury_7040863':'excluded from compact elementary selection','Mercury_SC_405062':'wind is not always available','MCAS_2015_5_11':'ambiguous compass energy wording','Mercury_SC_402064':'gas production can change factors besides pressure'},'review':'open adaptations are unreviewed; exact source-choice text is a lower bound, unmatched answers require semantic review'},'blimp':{'uids':uids,'pair_ids':'see individual records; deterministic hash ordering'}}}
    man['selected_files']={n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ['public-popqa.jsonl','public-arc.jsonl','public-blimp.jsonl']}
    (D/'source-manifest.json').write_text(json.dumps(man,indent=2,ensure_ascii=False)+'\n')
    return selected + arc + blimp
