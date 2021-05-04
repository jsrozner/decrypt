Repository for Decryping Cryptic Crosswords

# Download data
```
git clone git@github.com:jsrozner/decrypt.git
cd decrypt
mkdir puzzles
python guardian_scrape.py --save_directory="./puzzles"
```

# Reproducing our splits
```python
from decrypt.guardian_load import (
  load_guardian_splits,               # naive random split
  load_guardian_splits_disjoint,      # answer-disjoint split
  load_guardian_splits_disjoint_hash  # word-initial disjoint split
)

## returns
#each of these returns a tuple of
# - soln to clue map (string to List of clues mapping to that soln): Dict[str, List[GuardianClue]
# - list of all clues (List[GuardianClue])
# - Tuple of three lists (the train, val, test splits), each is List[GuardianClue]
# tuple of three lists are the splits for use in training a model
# 
# load_guardian_splits() will verify that
# - total glob length matches the one in paper (ie. number of puzzles)
# - total clue set length matches the one in paper
# - one of the clues in our train set matches our train set (i.e. a single clue
# spot check)
# if you get an assertion error or an exception during load, please file an
# issue, since the splits should be identical
# alternatively, if you don't care, you can pass `verify=False` to
# `load_guardian_splits`

soln_to_clue_map, all_clues_list, (train, val, test) = load_guardian_splits("./puzzles")
```

