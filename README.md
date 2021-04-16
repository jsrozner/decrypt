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

soln_to_clue_map, all_clues_list, (train, val, test) = load_guardian_splits("./puzzles")
```

