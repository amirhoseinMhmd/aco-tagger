# pos-aco

Reference implementation of **ACO-tagger** — Mohammadi, Hajiaghajani & Bahrani,
*"A Novel Method for Part-of-Speech Tagging using Ant Colony Optimization"*
([arXiv:2303.16760](https://arxiv.org/abs/2303.16760)).

A sentence is tagged with a first-order hidden Markov model: each word has an
emission probability per tag, each tag pair a transition probability, and the
first word an initial distribution. Probabilities become additive edge costs,
which turns "best tagging" into "cheapest path through a trellis" — a path an
ant colony can search.

See [Fidelity to the paper](#fidelity-to-the-paper) for what matches, what the
paper leaves ambiguous, and two findings that affect how its headline result
should be read.

No third-party dependencies to tag. Python 3.10+. (Training from the bundled
NLTK corpora needs `nltk`; a model loaded from CSV never does.)

## Usage

```bash
python -m pos_aco "the company said it will sell the unit"   # ant colony
python -m pos_aco "the company said it will sell the unit" --solver viterbi
python -m pos_aco "john is dead" --data-dir data/toy         # 3-word sample
python -m unittest discover                                  # 176 tests
```

```
the/DT company/NN said/VBD it/PRP will/MD sell/VB the/DT unit/NN
cost: 13.6343
```

`--help` lists the colony knobs: `--ants`, `--generations`, `--alpha`
(pheromone weight), `--beta` (cost weight), `--evaporation`, `--deposit`, `--q`.

## Training a model

```bash
python -m pos_aco.train --corpus treebank --out data/treebank --evaluate both
python -m pos_aco.train --corpus mycorpus.txt --out data/mine
```

`--corpus` takes either `treebank` (the Penn Treebank sample shipped with NLTK)
or a path to a plain file with one sentence per line in `word/TAG word/TAG`
form. Probabilities are maximum-likelihood counts, with two adjustments:

- **add-k smoothing** on transitions and the initial distribution, so a tag pair
  the corpus happens not to contain is unlikely rather than impossible;
- **`<UNK>` from hapax legomena** — training words seen once become a single
  unknown-word entry. Half of any corpus's vocabulary occurs exactly once, so
  this is what lets the tagger handle words it has never met.

Emissions are deliberately *not* smoothed: that would fill in all
`vocabulary × tags` cells and destroy the sparsity the lexicon relies on.

Held-out accuracy on the treebank (391 sentences reserved, 40 PTB tags,
4,004 scored tokens):

| cost function | solver | overall | known | unknown |
| --- | --- | --- | --- | --- |
| paper eq. (1) | Viterbi (exact) | 81.12% | 88.70% | 34.46% |
| paper eq. (1) | ant colony | **82.87%** | 91.87% | 27.50% |
| log-likelihood | Viterbi (exact) | **88.29%** | 94.80% | 48.21% |
| log-likelihood | ant colony | 85.61% | 93.41% | 37.68% |

Under equation (1) the colony beats exact Viterbi, reproducing the paper's
headline in kind. Under the standard cost the ordering flips to the expected
one. [Below](#2-the-aco-beats-viterbi-result-is-reproducible-but-not-for-the-reason-it-looks-like)
explains why.

## Model data

Three CSVs per model directory (`data/toy`, `data/treebank`):

| file | shape | meaning |
| --- | --- | --- |
| `metadata.csv` | `key,value` rows | which tokenizer trained it (optional) |
| `initial.csv` | one column per tag, a single row | `P(tag)` for the first word |
| `transitions.csv` | the same columns, one row per previous tag | `P(tag \| previous tag)` |
| `lexicon.csv` | `word,tag,probability`, non-zero rows only | `P(word \| tag)` |

`initial.csv`'s header is the authoritative tag order; the other two are
validated against it, so a reordered or renamed column fails loudly instead of
silently mistagging.

The lexicon is stored in long form because a real corpus fills almost none of
the grid: the treebank model has 5,044 words × 40 tags = 201,760 cells but only
6,718 non-zero entries (3.3%). That is 181 KB instead of roughly 1.6 MB.

## Layout

```
text ──tokenizer──▶ words ──trellis──▶ costs ──solver──▶ path ──tagger──▶ tags
```

The package is layered, and **a layer may only import the ones above it**:

```
src/
├── domain/          pure: no I/O, no randomness, stdlib only
│   ├── model.py         the HMM data and its invariants
│   ├── trellis.py       probabilities → a (step, previous_tag, tag) cost lookup
│   ├── solver.py        the PathSolver protocol and Solution
│   ├── tokenizer.py     the one language-specific module: Tokenizer implementations
│   └── corpus.py        what a tagged corpus is: TaggedSentence, split()
├── search/          ← domain
│   ├── viterbi.py       exact dynamic-programming solver
│   ├── pheromone.py     trail storage + interchangeable deposit strategies
│   ├── ant.py           one ant's walk across one sentence
│   └── colony.py        the generational ACO loop
├── application/     ← domain, search
│   ├── tagger.py        composes a model with a solver
│   ├── training.py      counting a corpus into probabilities
│   ├── evaluation.py    held-out accuracy, split by known/unknown
│   └── ports.py         what it needs the outside world to provide
├── adapters/        ← domain, application (implements the ports)
│   ├── csv_model.py     CSV persistence
│   └── corpora.py       CoNLL-U, word/TAG files, NLTK
├── cli.py           the tagging command line
└── train.py         the training command line
```

`tests/test_architecture.py` walks the source with `ast` and **fails the build**
if that direction is ever violated — so the diagram is a property of the code,
not a claim about it. It also asserts the domain imports no `csv` or `pathlib`,
and that only `adapters` may touch NLTK.

It earned its keep immediately: on its first run it caught `application` pulling
`TaggedSentence` out of `adapters.corpora` rather than `domain.corpus`.

Two modules split along a layer boundary during this move, which is how you
know the boundary was real:

- `corpora.py` held both the `TaggedSentence` type (needed by `training` and
  `evaluation`) and the file readers. The type went to `domain/corpus.py`, the
  readers stayed in `adapters/`.
- `loading.py` held both the `LanguageModelLoader` protocol and its CSV
  implementation. The protocol went to `application/ports.py`, the
  implementation became `adapters/csv_model.py`.

Design notes:

- **Single responsibility.** Loading, cost construction, search and presentation
  are separate modules, now in separate layers. The old code did all four in
  `main.py`, with a second half-finished copy in `test.py`.
- **Dependency inversion, structurally.** `application/ports.py` declares what
  the use cases need; `adapters/` implements it. The dependency arrow points
  inward even though the data flows outward, so nothing in `domain`,
  `search` or `application` knows that CSV exists.
- **Dependency inversion.** `PosTagger` depends on the `PathSolver` protocol, so
  the exact and approximate solvers are interchangeable — which is what lets the
  tests check the colony against Viterbi.
- **Open/closed.** Reinforcement schemes are `DepositStrategy` objects
  (`AntCycle`, `AntDensity`, `AntQuantity`) rather than an `if strategy == 1`
  chain inside the ant.
- **Open/closed, again.** The cost function is injected into `TrellisBuilder`,
  so the paper's equation (1) and the standard log-likelihood coexist and are
  swapped with one flag rather than an edit.
- **KISS.** pandas and numpy were dropped; the stdlib `csv` module covers the
  whole need. Randomness is injected as a `random.Random`, so seeded runs are
  reproducible.
- **One definition of normalization.** Training and tagging both call
  `tokenizer.normalize`. If they diverged, a model trained on `Ali` could never
  be looked up for `ali` — the classic train/serve skew bug.
- **NLTK stays optional.** It is imported lazily inside `corpora.py`, so it is a
  training-time convenience and never a runtime dependency of the tagger.

## Language support

The model, trellis and solvers treat words and tags as opaque strings, so a new
language means **retraining, not rewriting**. Everything language-specific is
confined to `tokenizer.py`.

```bash
# Any Universal Dependencies treebank, for 100+ languages
python -m pos_aco.train --corpus UD_Persian-PerDT/fa_perdt-ud-train.conllu \
                        --out data/fa --tokenizer persian
python -m pos_aco "امروز هوا سرد است." --data-dir data/fa
```

`--corpus` accepts CoNLL-U (auto-detected by suffix, `--tagset upos|xpos`), a
plain `word/TAG` file, or `treebank`. The chosen tokenizer is written into the
model's `metadata.csv` and read back automatically when tagging, so a Persian
model is never queried with English normalisation.

| tokenizer | for | what it adds |
| --- | --- | --- |
| `default` | any whitespace-delimited language | strips punctuation by Unicode *category*, so `،` `؛` `。` `।` `«` behave like `.` and `,`; case-folds |
| `persian` | Persian / Farsi | folds Arabic↔Persian kaf, yeh, teh marbuta, hamza forms and digits; drops tatweel and harakat; keeps ZWNJ |
| `turkish` | Turkish / Azerbaijani | locale-correct dotted/dotless I (`ISTANBUL` → `ıstanbul`, not `istanbul`) |

Why the Persian one matters: Arabic kaf (U+0643) and Persian keheh (U+06A9) are
different codepoints for the same letter, and Unicode NFC does not merge them.
Real Persian text mixes both. On a corpus spelling *ketab* each way, the default
tokenizer produced **17** vocabulary entries where `persian` produced **16** —
the same word counted twice, splitting its counts and inflating the OOV rate.

### Known limits

- **Languages without whitespace** — Chinese, Japanese, Thai, Khmer, Lao — need a
  word segmenter in front of this. `DefaultTokenizer` returns the whole sentence
  as one token, and `test_languages.py` records that so it is not mistaken for a
  bug.
- **Morphologically rich languages** (Turkish, Finnish, Persian, Arabic) have
  high out-of-vocabulary rates, and the `<UNK>` entry is a single distribution
  with no suffix features. A word-level first-order HMM is structurally weak
  where the tag lives in the morphology.
- **Right-to-left scripts** tag correctly; only the `word/TAG` terminal output
  reads awkwardly under bidi.

## Fidelity to the paper

Run the paper's configuration with:

```bash
python -m pos_aco "..." --cost paper --preset paper
```

`--cost paper` is the default. `--preset paper` applies Table 4
(generations 3, ants 20, α 0.9, β 0.9, ρ 0.95, Q 10). `tests/test_paper.py`
asserts the paper's own numbers — the page-5 distances and Table 4 — so the
implementation is pinned to the published method rather than to a plausible
reimplementation of it.

Matching as specified: the trellis of figure 1 (one segment per word, nodes per
tag); η = 1/D; the selection rule of equation (2); and the evaporation of
equation (3), `c ← (1-ρ)c + ΣΔc`.

Four things are worth recording.

### 1. Equation (1) is not the HMM log-likelihood

The paper defines `D = emission ** log10(transition)`, verified here against
all three worked examples on page 5. That is a different function from the
`-log10(emission × transition)` the original code computed, and it has a
degeneracy: since `x ** 0 == 1`, an edge whose emission probability is 1 costs
exactly 1 however improbable its transition is.

| emission | transition | eq. (1) | log-likelihood |
| --- | --- | --- | --- |
| 1.0 | 0.5 | 1.0000 | 0.30 |
| 1.0 | 0.001 | 1.0000 | 3.00 |
| 1.0 | 1e-9 | **1.0000** | 9.00 |

So an unambiguous word ignores the transition model — the mechanism an HMM
relies on to disambiguate. The range also explodes: because the cost is
`10 ** (log10(e) · log10(t))`, a product of surprisals in the exponent, finite
edge costs on one real sentence spanned 1.31 to 9.1 × 10¹¹, letting a single
edge dominate a path sum.

### 2. The "ACO beats Viterbi" result is reproducible, but not for the reason it looks like

Viterbi is exact for *any* additive edge cost, equation (1) included. On 100
held-out sentences scored with equation (1), Viterbi reached strictly lower
cost on 74 and was **never once beaten** (colony lower on 0, tied on 26).

Yet the colony scores *higher accuracy* under that same cost (83.38% vs
81.42%). Both facts together say the objective is misaligned with correctness:
optimising equation (1) *better* makes tagging *worse*. The colony's advantage
comes from failing to optimise it, not from searching well. This is pinned by
`ExactSolverDominanceTest`.

### 3. Table 2's orientation is ambiguous, and the printed reading is unusable

Read with rows as the previous tag, `ADJ→V` and `V→DELM` are both 0 — and the
example sentence needs both. All 3,125 candidate taggings then cost infinity.
Transposed, exactly 4 are finite and the cheapest is the intended
*"today the weather is snowy"* (ADV N ADJ V DELM, cost 5.1261).

The page-5 examples cannot settle this: every second-segment case is either
emission = 1 (D = 1 whatever the transition) or emission = 0 (inf whatever it
is). The fixture in `tests/support.py` therefore transposes the printed table,
with `PrintedTransitionTableTest` recording why.

### 4. Smaller gaps

- The paper states the pheromone array starts at zero, which makes equation (2)
  a 0/0 in the first generation. `PheromoneTable` requires a positive initial
  level.
- Equation (3) sums an unspecified `Δc`, so Q = 10 needs pairing with a choice
  of `DepositStrategy`; `--deposit` selects one (default `cycle`).
- The corpus is Bijankhan (Persian, 2.6M words, 32 tags), which needs a licence
  request. The Penn Treebank model here is a runnable substitute, so the
  accuracy figures above are **not** comparable to the paper's 96.867%.

## What changed from the original

The original `ACO` class ran, but its pheromone feedback did nothing — it was
effectively a randomised greedy search that only found the right answer because
27 candidate taggings fit easily inside 1000 ant-runs. Fixed here:

- `_update_pheromone` indexed a 3-D array with 2-D logic and added a `list` into
  a float slot; it avoided a `TypeError` only because the example was 3×3×3.
- `_update_pheromone_delta` wrote `delta[i][j]` using *tags* as indices while the
  array was laid out `[step][previous][current]`, so deposits landed on unrelated
  edges. Its `Q / edge_cost` branch also shadowed its own loop variable.
- `Graph.rank` meant "number of words" but was used as "number of tags", and
  `allowed` was reset to a hardcoded `range(3)`. Both worked only because the
  sample sentence had as many words as the tag set had tags.
- The `allowed`/`tabu` bookkeeping was a travelling-salesman leftover: in tagging
  every tag stays available at every word, so it is gone.
- Padding the first word's costs with `inf` rows is replaced by storing the
  start costs separately, behind a uniform `cost(step, previous, tag)` lookup.
- Results were returned as a `dict` keyed by word, which dropped a tag whenever a
  word repeated ("is ali is"). They are now parallel tuples.

`tests/test_solvers.py` pins the fix down: with `--beta 0` the cost heuristic is
switched off, so only pheromone can guide the ants — and the trail on the optimal
edges ends up more than 100× the trail on its rivals.
