"""Deterministic, original, short English diagnostic items; no random selection."""
import json
from collections import Counter
from pathlib import Path

SIZES = {'dev': 1, 'test': 2, 'reserve': 1}
# Each split uses a separate vocabulary and arithmetic range. The task structures
# are deliberately shared, as documented in the manifest.
NOUNS = {
 'dev': ['apples','buttons','cards','cups','drums','envelopes','feathers','gloves','hats','jars','keys','lamps'],
 'test': ['maps','notebooks','oranges','pencils','quilts','ribbons','shells','tickets','umbrellas','vases','whistles','zippers'],
 'reserve': ['anchors','baskets','candles','dishes','erasers','flags','gates','hooks','islands','jewels','kites','lanterns'],
}
PLACES = {
 'dev': ['attic','bakery','cabin','dock','engine room','farm','garden','hall','inn','jetty','kitchen','library'],
 'test': ['market','nursery','orchard','pantry','quay','roof','shed','tower','underpass','valley','workshop','yard'],
 'reserve': ['archive','bridge','cellar','depot','entrance','factory','gallery','harbor','infirmary','junction','kennel','lobby'],
}
WORDS = {
 'dev': ['amber','blue','coral','dull','eager','fair','green','happy'],
 'test': ['ivory','jade','kind','light','merry','navy','olive','pink'],
 'reserve': ['quiet','red','soft','teal','urban','vivid','warm','yellow'],
}

def build_authored(output_dir):
    """Write three JSONL splits and manifest under output_dir; return all cases.

    output_dir is a data directory, created if needed. Existing authored files
    are replaced deterministically; no other file is touched.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    def add(split, category, family, prompt, answer, constraints=None, mode='exact', **extra):
        idx = sum(c['split'] == split for c in cases)
        row = {'id': f'authored-{split}-{idx:03d}', 'split': split,
               'family_id': f'{split}-{family}', 'category': category,
               'source': {'kind': 'authored', 'version': '1'},
               'max_new_tokens': 48 if category == 'instructions' else 32,
               'messages': prompt if isinstance(prompt, list) else [{'role':'user','content':prompt}],
               'gold': [answer], 'content_mode': mode, 'constraints': constraints or []}
        row.update(extra)
        cases.append(row)
    for split, multiplier in SIZES.items():
        n = 4 * multiplier
        objects = NOUNS[split]
        places = PLACES[split]
        words = WORDS[split]
        offset = {'dev': 0, 'test': 10, 'reserve': 20}[split]
        # Four verbal forms: entailment with a named object; ordered finish;
        # exclusive assignment; and chain of container membership.
        for i in range(n):
            x, y, z = objects[i], objects[i+1], objects[i+2]
            polarity = i % 2 == 0
            statement = 'tagged' if polarity else 'untagged'
            question = {'dev': f'Every item in the red box is {statement}. The {x} are in the red box. Are the {x} tagged?', 'test': f'The {x} are in the red box. No item in the red box is tagged.' if not polarity else f'The {x} are in the red box. Every item in that box is tagged.', 'reserve': f'The red box contains the {x}. All items in it are {statement}.'}[split]
            if split != 'dev': question += f' Are the {x} tagged?'
            add(split,'reasoning',f'verbal-class-{i}',question+' Answer yes or no.','yes' if polarity else 'no',subtask='verbal',difficulty='direct')
            last = i % 2 == 1
            order = f'The {z} were packed after the {y}; the {y} were packed after the {x}.' if split == 'test' else f'The {x} were packed before the {y}. The {y} were packed before the {z}.'
            add(split,'reasoning',f'verbal-order-{i}',order+f' Which were packed {"last" if last else "first"}? Answer with the item name.',z if last else x,subtask='verbal',difficulty='composed')
            add(split,'reasoning',f'verbal-assignment-{i}',f'Exactly one of the {x} and the {y} is wrapped. The {x} are not wrapped. Which are wrapped? Answer with the item name.',y,subtask='verbal',difficulty='direct')
            add(split,'reasoning',f'verbal-container-{i}',f'The {x} are inside a crate. That crate is inside the {places[i]}. Where are the {x}? Answer with the place only.',places[i],subtask='verbal',difficulty='composed')
        # Fixed cardinal directions make reference frame explicit.
        for i in range(n):
            x, y = places[i], places[i+1]
            horizontal = i % 2 == 0
            spatial = (f'The {x} is directly west of the {y}. Which place is on the right?' if horizontal else f'The {x} is directly south of the {y}. Which place is lower?')
            add(split,'reasoning',f'spatial-east-{i}',f'On a map, north is up and east is right. {spatial} Answer with the place only.',y if horizontal else x,subtask='spatial',difficulty='direct')
            north = i % 2 == 1
            if split == 'reserve':
                chain = f'The {places[i+2]} is directly south of the {y}. The {y} is directly south of the {x}.'
            else:
                chain = f'The {x} is directly north of the {y}. The {y} is directly north of the {places[i+2]}.'
            add(split,'reasoning',f'spatial-chain-{i}',f'On a map, north is up and east is right. {chain} Which place is farthest {"north" if north else "south"}? Answer with the place only.',x if north else places[i+2],subtask='spatial',difficulty='composed')
        for i in range(n):
            a = offset+i+3; b = i+2
            add(split,'reasoning',f'arithmetic-add-{i}',f'A box contains {a} {objects[i]}. {b} more are put in. How many are in the box? Answer with a number.',str(a+b),subtask='arithmetic',difficulty='direct')
            add(split,'reasoning',f'arithmetic-two-step-{i}',f'A box contains {a} {objects[i]}. {b} more are put in, then 1 is removed. How many remain? Answer with a number.',str(a+b-1),subtask='arithmetic',difficulty='composed')
        # These supplied facts avoid factual recall; two prose families require
        # independent review while their formal constraints remain automatic.
        for i in range(n):
            w=words[i]; thing=objects[i]; place=places[i]
            phrase=f'The {thing} are in the {place}.'
            add(split,'instructions',f'instr-copy-{i}',f'Copy exactly, with the same punctuation: {phrase}',phrase,[{'type':'raw_exact','value':phrase}],instruction_family='copy')
            add(split,'instructions',f'instr-required-{i}',f'The {thing} are in the {place}. In one short English sentence, say where they are. Include the word {place}.',phrase,[{'type':'required_word','value':place}],mode='review',reference_rubric=f'States that the {thing} are in the {place}; no contradictory claim.',instruction_family='required_word')
            add(split,'instructions',f'instr-forbidden-{i}',f'The {thing} are in the {place}. In one short English sentence, say where they are. Do not use the word {w}.',phrase,[{'type':'forbidden_word','value':w}],mode='review',reference_rubric=f'States that the {thing} are in the {place}; no contradictory claim.',instruction_family='forbidden_word')
            add(split,'instructions',f'instr-lower-{i}',f'Write this word in lowercase and nothing else: {w.upper()}',w,[{'type':'lowercase'}],instruction_family='lowercase')
            add(split,'instructions',f'instr-upper-{i}',f'Write this word in uppercase and nothing else: {w}',w.upper(),[{'type':'uppercase'}],instruction_family='uppercase')
            bullets=f'- {thing}\n- {place}'
            add(split,'instructions',f'instr-bullets-{i}',f'Write exactly two bullet lines: first {thing}, then {place}. Begin each line with a hyphen and a space.',bullets,[{'type':'bullet_lines','count':2,'prefix':'- '}],instruction_family='bullet_lines')
            ending=f'{w}.'
            add(split,'instructions',f'instr-ending-{i}',f'Write only this sentence, including its final period: The label is {w}.',f'The label is {ending}',[{'type':'ends_with','value':ending}],instruction_family='ending')
            composite=f'{w.upper()}\n{thing.upper()}'
            add(split,'instructions',f'instr-composed-{i}',f'Write two bullet lines, first {w}, then {thing}. Use uppercase letters and begin each line with a hyphen and a space.',f'- {w.upper()}\n- {thing.upper()}',[{'type':'uppercase'},{'type':'bullet_lines','count':2,'prefix':'- '}],instruction_family='composition')
        # Six or twelve paired scenarios per type. Each pair varies only the
        # relevant fact/mapping/preference and has a changed canonical answer.
        for i in range(6*multiplier):
            thing=objects[i]; p=places[i]; q=places[(i+1)%len(places)]
            pair=f'{split}-reading-{i}'
            for v,loc in enumerate((p,q)):
                distractor=objects[(i+3)%len(objects)]
                other=places[(i+4)%len(places)]
                passage = {'dev': f'The inventory says the {distractor} are in the {other}. The {thing} are stored in the {loc}.', 'test': f'Two entries appear in a store log: {distractor} belong in the {other}; {thing} belong in the {loc}. The log was checked this morning.', 'reserve': f'A keeper checks the {loc} and finds the {thing} there. In a separate check, the {distractor} are found in the {other}.'}[split]
                add(split,'context',pair,passage+f' Where are the {thing}, according to this passage? Answer with the place only.',loc,pair_id=pair,variant=('a' if v == 0 else 'b'),context_type='reading')
            pair=f'{split}-fiction-{i}'
            # Explicit invented setting, no dependence on actual geography.
            for v,label in enumerate(('silver','orange') if i%2==0 else ('orange','silver')):
                add(split,'context',pair,f'In the fictional world of Story {split} {i}, the sky is {label} at noon. In that world, what color is the sky at noon? Answer with the color only.',label,pair_id=pair,variant=('a' if v == 0 else 'b'),context_type='fiction')
            pair=f'{split}-examples-{i}'
            # Mapping transfer and separately inferred ordering transformations.
            a,b=words[i%8],words[(i+1)%8]
            base_labels={'dev':('sun','moon'), 'test':('star','cloud'), 'reserve':('rain','snow')}[split]
            labels=base_labels if i%2==0 else base_labels[::-1]
            for v in range(2):
                if i % 2 == 0:
                    la,lb=labels if v==0 else labels[::-1]
                    prompt=(f'Examples:\n{a} -> {la}\n{b} -> {lb}\n'
                            f'For two words, write their matching outputs in the same order, separated by one space.\n'
                            f'{b} {a} ->')
                    answer=f'{lb} {la}'
                    learning='mapping_transfer'
                else:
                    # Each example contains distinct words. Two demonstrations
                    # distinguish preserve-order from reverse-order; the test
                    # combination is unseen, with its words also in examples.
                    c,d=words[(i+2)%8],words[(i+3)%8]
                    def transform(x,y):
                        return f'{x} {y}' if v == 0 else f'{y} {x}'
                    prompt=(f'Complete the last line by following the same pattern as the examples. '
                            f'Write only the two resulting words.\n{a} {b} -> {transform(a,b)}\n'
                            f'{c} {d} -> {transform(c,d)}\n{a} {d} ->')
                    answer=transform(a,d)
                    learning='inferred_order'
                add(split,'context',pair,prompt,answer,pair_id=pair,variant=('a' if v == 0 else 'b'),context_type='examples',example_type=learning)
            pair=f'{split}-dialogue-{i}'
            # A correction in both variants, with same final user message.
            for v,chosen in enumerate((p,q)):
                initial=q if v==0 else p
                messages=[{'role':'user','content':f'For our fictional delivery, use the {initial}.'},
                          {'role':'assistant','content':'I will use that place.'},
                          {'role':'user','content':f'Correction: use the {chosen} instead.'},
                          {'role':'assistant','content':'I have noted the correction.'},
                          {'role':'user','content':'Which place should be used? Answer with the place only.'}]
                add(split,'context',pair,messages,chosen,pair_id=pair,variant=('a' if v == 0 else 'b'),context_type='dialogue')
        for i in range(n):
            thing=objects[i]; place=places[i]; word=words[i]
            add(split,'stopping',f'stop-copy-{i}',f'Copy this word and then stop: {word}',word,stopping_type='copy')
            add(split,'stopping',f'stop-fact-{i}',f'The {thing} are in the {place}. Where are they? Give only the place.',place,stopping_type='fact')
            add(split,'stopping',f'stop-list-{i}',f'List these two words in order on one line, separated by a comma and a space: {word}, {thing}.',f'{word}, {thing}',stopping_type='list')
            add(split,'stopping',f'stop-dialogue-{i}',[{'role':'user','content':f'My chosen label is {word}.'},{'role':'assistant','content':'I have noted it.'},{'role':'user','content':'What label did I choose? Answer with one word.'}],word,stopping_type='dialogue')
    manifest={'version':1,'generator':'kefir_bench.authored.build_authored','split_counts':dict(Counter(c['split'] for c in cases)),
      'split_inventory':{s:{'reasoning':{'verbal':16*m,'spatial':8*m,'arithmetic':8*m},'instructions':32*m,'context_pairs':{'reading':6*m,'fiction':6*m,'examples':6*m,'dialogue':6*m},'stopping':16*m} for s,m in SIZES.items()},
      'isolation':'Pairs, vocabularies and specific scenarios do not cross splits. Some wording and passage structures vary between splits; task mechanisms and several templates repeat across splits. Parameter changes are not independent templates. These are held-out instances, not a clean held-out task-structure evaluation.',
      'example_learning':'Half of example pairs transfer a two-item mapping with an explicit composition instruction. Half infer whether to preserve or reverse the order of a new pair from two demonstrations. These examples distinguish the two specified transformations but cannot logically exclude every possible alternative rule.',
      'review':'Required-word and forbidden-word prose cases require semantic review against reference_rubric. Other text golds use conservative exact matching. References are program-generated and have not received independent human adjudication.',
      'limitations':'Synthetic short prompts, small vocabulary, repeated templates and correlated items limit generalization. Family IDs identify individual scenarios, not statistically independent templates. Answer aliases beyond the canonical answer may require review. The panel is exposed and is not sealed.'}
    for split in SIZES:
        with (output_dir/f'authored-{split}.jsonl').open('w',encoding='utf-8') as f:
            for row in cases:
                if row['split']==split: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    (output_dir/'authored-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return cases
