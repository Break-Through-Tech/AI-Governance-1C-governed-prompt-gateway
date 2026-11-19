"""Build a simulated customer-service workload (issue #30).

Issue #19 measured compression on raw user messages alone and reached only
0.2% reduction, because its test prompts average ~34 tokens with nothing
else around them. In a real deployment, most of a request's tokens come
from the system prompt, retrieved FAQ/policy context, and chat history --
not the user's one-line question. This wraps real safe test-set queries in
a simulated version of all three, so compression can be measured against a
realistic token budget instead.

The workload's design (system prompt text, context pool, history template,
sample size, selection seed) is fixed here before any measurement is taken
(scripts/evaluate_workload_compression.py), so the design isn't tuned to
the result -- per issue #30's requirement.

Every component below is authored, fixed text, not real retrieval or a
real conversation log. "Retrieval" is simulated by seeded random sampling
from a small fixed FAQ/policy pool; "chat history" is one fixed template
reused for every query, representing a typical prior-conversation length,
not real per-user history.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEST_INPUT = ROOT / 'data/test_clean.csv'
OUTPUT_DIR = ROOT / 'data/processed/workload'

SAMPLE_SIZE = 150
SEED = 42

SYSTEM_PROMPT = (
    "You are a customer support assistant for a mid-size e-commerce company. "
    "Your job is to help customers with questions about their orders, shipping, "
    "returns, refunds, billing, and account issues. Always be polite, patient, "
    "and professional, even if the customer is frustrated. Do not make promises "
    "about refunds or exceptions to policy unless the provided policy context "
    "explicitly allows it. If you are not sure about something, say so honestly "
    "rather than guessing. Keep your answers clear and concise, and avoid use of "
    "jargon the customer may not understand. When relevant, reference the "
    "specific policy or FAQ information provided to you rather than making up "
    "details. Never ask the customer for full payment card numbers, passwords, "
    "or other sensitive account credentials in chat. If a request falls outside "
    "what you can help with, let the customer know how to reach a human agent."
)

# Same instructions, condensed -- authored once, not algorithmically derived.
TIGHTENED_SYSTEM_PROMPT = (
    "You are an e-commerce customer support assistant. Help with orders, "
    "shipping, returns, refunds, billing, and account issues. Be polite and "
    "professional. Only grant refunds or exceptions if the provided policy "
    "explicitly allows it. If unsure, say so. Be clear and concise. Use only "
    "the provided policy/FAQ context; never invent policy details. Never "
    "request full card numbers, passwords, or credentials. If you can't help, "
    "direct the customer to a human agent."
)

# A small fixed pool a retriever might pull FAQ/policy snippets from.
# "Retrieval" per query is a seeded random sample from this pool, since
# this workload tests compression, not retrieval quality.
CONTEXT_POOL = [
    ("shipping_policy", "Standard shipping takes 5-7 business days within the "
     "continental US. Expedited shipping (2-3 business days) is available at "
     "checkout for an additional fee. Orders placed after 2pm ET ship the next "
     "business day. We do not currently ship internationally."),
    ("returns_policy", "Items can be returned within 30 days of delivery for a "
     "full refund, provided they are unused and in their original packaging. "
     "Final sale items, gift cards, and personalized products cannot be "
     "returned. Return shipping is free for defective or incorrect items."),
    ("refund_policy", "Refunds are issued to the original payment method within "
     "5-10 business days of the returned item being received at our warehouse. "
     "Store credit is issued immediately if the customer selects that option "
     "instead of a refund to the original payment method."),
    ("account_policy", "Customers can reset their password from the sign-in "
     "page by selecting 'Forgot password' and following the emailed link. "
     "Account email addresses can be changed in Account Settings after "
     "verifying the new address. We never ask for passwords over chat."),
    ("billing_policy", "We accept major credit cards, PayPal, and store gift "
     "cards. Subscriptions are billed monthly on the signup date and can be "
     "cancelled anytime from Account Settings with no cancellation fee. "
     "Failed payments are retried automatically up to three times."),
    ("warranty_policy", "Most products carry a 1-year manufacturer warranty "
     "covering defects in materials or workmanship. Warranty claims require "
     "proof of purchase and are handled by requesting a replacement or repair "
     "through Support; normal wear and accidental damage are not covered."),
]
CONTEXT_PER_QUERY = 4       # simulated "retrieved" snippets, ranked
CONTEXT_KEPT_AFTER_COMPRESSION = 2  # kept: the top-N by simulated retrieval rank

# One fixed prior conversation, reused for every query -- represents a
# typical history length before the real question, not a real per-user log.
CHAT_HISTORY = [
    ('customer', "Hi, I need some help with my account."),
    ('assistant', "Hello! I'd be happy to help with your account. Could you "
                  "tell me a bit more about what's going on?"),
    ('customer', "I placed an order last week and I'm not sure if it shipped yet."),
    ('assistant', "Let me check on that for you. Can you confirm the order "
                  "number or the email address used for the order?"),
    ('customer', "Sure, it's linked to my account email."),
    ('assistant', "Got it, thanks. I can see your order in our system. It "
                  "looks like it's currently being processed."),
    ('customer', "Okay, that's good to know. I actually have another question too."),
    ('assistant', "Of course! Go ahead, I'm happy to help with that as well."),
]
HISTORY_TURNS_KEPT_AFTER_COMPRESSION = 2  # last customer+assistant exchange only


def select_queries(test_input, sample_size, seed):
    df = pd.read_csv(test_input, encoding='utf-8-sig')
    if list(df.columns) != ['user_input', 'label']:
        raise ValueError('Expected exactly user_input,label columns in test_clean.csv')
    safe = df[df['label'] == 'safe'].reset_index().rename(columns={'index': 'test_row'})
    if len(safe) < sample_size:
        raise ValueError(f'Only {len(safe)} safe rows available, need {sample_size}')
    sampled = safe.sample(n=sample_size, random_state=seed).sort_values('test_row')
    return sampled[['test_row', 'user_input']].to_dict('records')


def build_workload(test_input=TEST_INPUT, output_dir=OUTPUT_DIR,
                    sample_size=SAMPLE_SIZE, seed=SEED):
    queries = select_queries(test_input, sample_size, seed)
    rng = random.Random(seed)
    context_ids = [snippet_id for snippet_id, _ in CONTEXT_POOL]

    records = []
    for row in queries:
        text = row['user_input']
        retrieved = rng.sample(context_ids, CONTEXT_PER_QUERY)  # simulated retrieval rank
        records.append({
            'test_row': int(row['test_row']),
            'prompt_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
            'user_message': text,
            'retrieved_context_ids': retrieved,
        })

    workload = {
        'design': {
            'sample_size': sample_size,
            'seed': seed,
            'source': str(test_input.relative_to(ROOT)) if test_input.is_relative_to(ROOT) else str(test_input),
            'source_label_filter': 'safe',
            'system_prompt': SYSTEM_PROMPT,
            'tightened_system_prompt': TIGHTENED_SYSTEM_PROMPT,
            'context_pool': dict(CONTEXT_POOL),
            'context_per_query': CONTEXT_PER_QUERY,
            'context_kept_after_compression': CONTEXT_KEPT_AFTER_COMPRESSION,
            'chat_history': CHAT_HISTORY,
            'history_turns_kept_after_compression': HISTORY_TURNS_KEPT_AFTER_COMPRESSION,
        },
        'queries': records,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / 'simulated_workload.json'
    out_path.write_text(json.dumps(workload, indent=2) + '\n', encoding='utf-8')
    return workload


def main():
    workload = build_workload()
    print(f"Built a simulated workload of {len(workload['queries'])} safe queries "
          f"(sample_size={workload['design']['sample_size']}, seed={workload['design']['seed']}).")
    print(f"Saved to {OUTPUT_DIR / 'simulated_workload.json'}")


if __name__ == '__main__':
    main()
