# Input schemas

All image arrays are preprocessed 96×96 cells in HWC layout, on the [0,255]
intensity scale (3 channels for BBBC/Allen, 5 for JUMP). Do not pass normalized
[-1,1] arrays; the loaders perform that conversion. Do not silently recompute
the splits. Same-batch/plate control cells must be present in each split.

`metadata.csv` must contain an initial CSV index column, followed by the fields
below. `SAMPLE_KEY` must be unique. `SPLIT` is `train` or `test`, with any existing
validation rows retained. BBBC/Allen accept STATE 0/1 or ctrl/trt; JUMP accepts
ctrl/trt, control/treated or 0/1, but its cross-plate statistics builder expects
the canonical lowercase **ctrl/trt** spelling.

| Dataset | Required fields |
|---|---|
| BBBC | SAMPLE_KEY, SPLIT, STATE, BATCH, CPD_NAME, ANNOT |
| JUMP | SAMPLE_KEY, SPLIT, STATE, PLATE, WELL, BROAD_SAMPLE, PERT_TYPE |
| Allen | SAMPLE_KEY, SPLIT, STATE, BATCH, CPD_NAME, STRUCTURE, ANNOT |

`ANNOT` is the BBBC MoA or Allen drug label (one stable label per condition).
For Allen, CPD_NAME indexes the **drug×structure combination**, not drug alone;
STRUCTURE is the structure/target label, and each BATCH has a single structure.

`embeddings.csv` has condition identifiers as its first/index column and numeric
feature columns. BBBC indexes CPD_NAME (1024 features); JUMP indexes BROAD_SAMPLE
(1224 features); Allen indexes CPD_NAME (12 features, additive 5-drug+7-structure
one-hot). Supply the same encoding/vocabulary order as your paired metadata.
Embeddings themselves are not bundled.

The dataset resolvers accept these storage layouts:

- BBBC: direct `images/<SAMPLE_KEY>.npy` or the CellFlux batch/well nesting
  implemented by `resolve_bbbc021_image`.
- Allen: direct `images/<SAMPLE_KEY>.npy` or `images/<plate>/<cell>.npy` for
  SAMPLE_KEY `<plate>__<cell>`.
- JUMP: `images/<plate>/<well>_<site>/<well>_<site>_<remainder>.npy` for
  SAMPLE_KEY `<plate>_<well>_<site>_<remainder>`. Use this canonical layout for
  both training and the TRAIN statistics builders.

BBBC held-out compounds are listed explicitly in each BBBC JSON configuration.
They are removed from treated rows, never from the control pool. Frozen feature
statistics are generated from TRAIN cells only; TEST images do not form training
prototypes. Data and feature caches are runtime inputs and are not bundled here.
