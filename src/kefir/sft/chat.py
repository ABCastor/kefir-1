"""Terminal chat with Kefir 1 SFT weights through the exact training renderer.

No download, telemetry or transcript persistence: the conversation lives in memory.
Sampling uses the core rules (greedy at temperature 0, else nucleus sampling); output is
restricted to ordinary text IDs plus EOD, and a reply ends at EOD or at the token cap.
"""
from __future__ import annotations

import math
import os
import sys

from .renderer import EOD, MAX_POSITIONS, ChatRenderer, ContextOverflow

# Neutral default; a named identity can be supplied with --system-prompt.
SYSTEM_PROMPT = "You are a helpful assistant."
KEFIR1_PARAMETERS = 144_074_648


def fit_context(renderer, messages, max_new_tokens, limit=MAX_POSITIONS):
    """Remove oldest whole exchanges; preserve System and the latest User."""
    kept = [dict(m) for m in messages]
    offset = int(bool(kept and kept[0]['role'] == 'system'))
    removed = 0
    while True:
        try:
            prefix = renderer.generation_prompt(kept)
        except ContextOverflow:
            prefix = None
        if prefix is not None and len(prefix) + max_new_tokens <= limit:
            return kept, prefix, removed
        if len(kept) - offset < 3:
            raise ValueError('The latest message plus System prompt exceeds the context budget. '
                             'Shorten it or use /reset.')
        del kept[offset:offset + 2]
        removed += 1


def generate_stream(model, prefix, *, max_new_tokens, seed, temperature, top_p, precision):
    """Yield one token at a time using the core sampling rules."""
    import torch
    from kefir.trainer.eval_generation import _autocast_context, _filter_logits
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    sequence = list(prefix)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            with _autocast_context(device, precision):
                logits = model(torch.tensor([sequence], device=device, dtype=torch.long))
            scores = logits[0, -1].float()
            if temperature == 0:
                token = int(scores.argmax())
            else:
                probabilities = torch.softmax(_filter_logits(scores[None] / temperature, top_k=0, top_p=top_p), dim=-1)
                if not torch.isfinite(probabilities).all() or bool((probabilities.sum(-1) <= 0).any()):
                    raise FloatingPointError('Invalid sampling probabilities')
                token = int(torch.multinomial(probabilities, 1, generator=generator).item())
            yield token
            if token == EOD:
                break
            sequence.append(token)


class Chat:
    """One in-memory conversation over a loaded model."""

    def __init__(self, model, *, system_prompt=SYSTEM_PROMPT, max_new_tokens=128,
                 temperature=0.0, top_p=0.95, seed=17):
        import torch
        from kefir.tokenizer import load_tokenizer
        from kefir.trainer.eval_generation import GenerationRequest, sample_token_ids_batched
        if not 1 <= max_new_tokens < MAX_POSITIONS:
            raise ValueError('max_new_tokens must be 1..2047')
        if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
            raise ValueError('invalid sampling parameters')
        self.renderer = ChatRenderer(load_tokenizer())
        self.system_prompt, self.max_new_tokens = system_prompt, max_new_tokens
        self.temperature, self.top_p, self.seed = temperature, top_p, seed
        device = next(model.parameters()).device
        self.precision = 'fp16' if device.type == 'cuda' else 'fp32'
        self.sample, self.request = sample_token_ids_batched, GenerationRequest

        class TextVocabulary(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, ids):
                return self.inner(ids)[..., :EOD + 1]

        self.text_model = TextVocabulary(model)
        self.reset()

    def reset(self):
        self.messages = ([{'role': 'system', 'content': self.system_prompt}]
                         if self.system_prompt.strip() else [])

    def reply(self, text, on_text=None, warn=lambda message: print(message, file=sys.stderr)):
        pending = self.messages + [{'role': 'user', 'content': text}]
        kept, prefix, removed = fit_context(self.renderer, pending, self.max_new_tokens)
        if removed:
            warn(f'[Removed {removed} oldest exchange(s) to stay within {MAX_POSITIONS} tokens.]')
        emitted = ''
        if on_text is None:
            ids = self.sample(self.text_model, [self.request(tuple(prefix), self.max_new_tokens, self.seed)],
                              max_seq_len=MAX_POSITIONS, eos_token_id=EOD, temperature=self.temperature,
                              top_p=self.top_p, top_k=0, precision=self.precision, batch_size=1)[0]
        else:
            from tokenizers import Tokenizer
            from tokenizers.decoders import DecodeStream
            from importlib.resources import files
            decoder = Tokenizer.from_str(files('kefir.assets').joinpath('superbpe_8000_v2.json').read_text())
            stream = DecodeStream(skip_special_tokens=False)
            ids = []
            for token in generate_stream(self.text_model, prefix, max_new_tokens=self.max_new_tokens,
                                         seed=self.seed, temperature=self.temperature,
                                         top_p=self.top_p, precision=self.precision):
                ids.append(token)
                if token == EOD:
                    break
                chunk = stream.step(decoder, token)
                if chunk:
                    emitted += chunk
                    on_text(chunk)
        ended = bool(ids and ids[-1] == EOD)
        body = ids[:-1] if ended else ids
        result = self.renderer.tokenizer.decode(body)
        if on_text is not None:
            if not result.startswith(emitted):
                raise ValueError('Streaming decoder disagrees with complete decoding')
            if len(result) > len(emitted):
                on_text(result[len(emitted):])
        if result.strip():
            self.messages = kept + [{'role': 'assistant', 'content': result}]
        if not ended:
            warn('[Reply reached the generation limit.]')
        return result


def read_message():
    """GNU readline keeps a bracketed multiline paste in one User message.

    /paste + /send is an explicit fallback for terminals without bracketed paste.
    """
    text = input('You> ')
    if text.strip() == '/paste':
        print('Paste the complete message; type /send on its own line to submit.')
        lines = []
        while True:
            line = input()
            if line.strip() == '/send':
                break
            lines.append(line)
        return '\n'.join(lines).strip()
    return text.strip()


def configure_terminal_input():
    try:
        import readline
        readline.parse_and_bind('set enable-bracketed-paste on')
        # Conversations stay in memory, never a readline history file.
        readline.set_history_length(0)
    except ImportError:
        pass


def main(argv=None):
    import argparse
    import torch
    from kefir.artifact import load_weights
    p = argparse.ArgumentParser(prog='kefir chat', description=__doc__)
    p.add_argument('--weights', required=True, help='Kefir 1 SFT .safetensors weights file')
    p.add_argument('--sha256', help='expected SHA-256 of the weights file')
    p.add_argument('--trusted-state', action='store_true',
                   help='--weights is a trainer state (.pt/.pkl) you trust to execute code')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top-p', type=float, default=.95)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--system-prompt', default=SYSTEM_PROMPT)
    p.add_argument('--no-stream', action='store_true', help='Print only after the entire response is generated')
    p.add_argument('--prompt', help='One response, then exit; otherwise interactive chat')
    a = p.parse_args(argv)
    if a.threads < 1:
        p.error('threads must be positive')
    try:
        device = torch.device(a.device)
    except RuntimeError:
        p.error(f'unknown device {a.device!r}; use cpu or cuda')
    if device.type == 'cuda' and not torch.cuda.is_available():
        p.error('--device cuda was given but CUDA is not available here; use --device cpu')
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    if a.device == 'cpu' and hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:a.threads])
    print('Loading Kefir 1 weights...', flush=True)
    model, config, _ = load_weights(a.weights, device=a.device, expected_sha256=a.sha256,
                                    trusted=a.trusted_state)
    if config.max_seq_len != MAX_POSITIONS or sum(x.numel() for x in model.parameters()) != KEFIR1_PARAMETERS:
        p.error('weights are not the Kefir 1 model (2048 context, 144,074,648 parameters)')
    try:
        chat = Chat(model, system_prompt=a.system_prompt, max_new_tokens=a.max_new_tokens,
                    temperature=a.temperature, top_p=a.top_p, seed=a.seed)
    except ValueError as error:
        p.error(str(error))

    def answer(text):
        print('Assistant> ', end='', flush=True)
        if a.no_stream:
            print(chat.reply(text) or '[Empty reply]', flush=True)
        else:
            reply = chat.reply(text, on_text=lambda chunk: print(chunk, end='', flush=True))
            if not reply:
                print('[Empty reply]', end='', flush=True)
            print(flush=True)
    if a.prompt is not None:
        answer(a.prompt)
        return 0
    configure_terminal_input()
    print('Ready. Paste multiline text as one message, then press Enter. /paste + /send also works.')
    print('/reset clears the conversation; /system shows the System prompt; /exit quits.')
    while True:
        try:
            text = read_message()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text in {'/exit', '/quit'}:
            break
        if text == '/reset':
            chat.reset()
            print('Conversation cleared.')
            continue
        if text == '/system':
            print(chat.system_prompt or '[No System prompt]')
            continue
        if not text:
            continue
        try:
            answer(text)
        except ValueError as error:
            print(str(error), file=sys.stderr)
    return 0


__all__ = ["Chat", "SYSTEM_PROMPT", "fit_context", "generate_stream", "main"]
