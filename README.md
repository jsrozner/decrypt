[comment]: <> (adapted from https://github.com/paperswithcode/releasing-research-code)

# Decrypting Cryptic Crosswords: Semantically Complex Wordplay Puzzles as a Target for NLP

This repository is the official implementation of 
[Decrypting Cryptic Crosswords: Semantically Complex Wordplay 
Puzzles as a Target for NLP](https://arxiv.org/abs/2104.08620). 
Please cite arxiv or [Neurips 2021 version](https://proceedings.neurips.cc/paper/2021/hash/5f1d3986fae10ed2994d14ecd89892d7-Abstract.html)

[comment]: <> (>📋  todo Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials)
The dataset is also available at https://doi.org/10.5061/dryad.n02v6wwzp

## Requirements

This will enable you to download and replicate the datasplits, but it has not been updated
to include all requirements to run the (baselines and experiments notebooks).
```setup
pip install -r requirements.txt
```

## Preparing data
```setup
git clone <anonymized>  # if using code supplement, just unzip
cd decrypt
pushd ./data && unzip "*.json.zip" && popd
```

### Download data (can safely be ignored)
If you want to download the data yourself from the web (you probably don't want to)
```setup
git clone <anonymized>  # if using code supplement, just unzip
cd decrypt
mkdir -p './data/puzzles'
python decrypt/scrape_parse/guardian_scrape.py --save_directory="./data/puzzles"
```
Then when you run `load_guardian_splits` you will run 
`load_guardian_splits("./data/puzzles", load_from_files=True, use_premade_json=False)`


# Reproducing our splits
```python
from decrypt.scrape_parse import (
  load_guardian_splits,               # naive random split
  load_guardian_splits_disjoint,      # answer-disjoint split
  load_guardian_splits_disjoint_hash  # word-initial disjoint split
)
from decrypt.scrape_parse.guardian_load import SplitReturn
"""
each of these methods returns a tuple of `SplitReturn`
- soln to clue map (string to List of clues mapping to that soln): Dict[str, List[BaseClue]
this enables seeing all clues associated with a given answer word
- list of all clues (List[BaseClue])
- Tuple of three lists (the train, val, test splits), each is List[BaseClue]

Note that
load_guardian_splits() will verify that
- total glob length matches the one in paper (ie. number of puzzles downloaded matches)
- total clue set length matches the one in paper (i.e. filtering is the same)
- one of the clues in our train set matches our train set (i.e. a single clue
spot check for randomness)
If you get an assertion error or an exception during load, please file an
issue, since the splits should be identical
Alternatively, if you don't care, you can pass `verify=False` to
`load_guardian_splits`
"""

soln_to_clue_map, all_clues_list, (train, val, test) = load_guardian_splits()
```

## Replicating our work
We make code available to replicate the entire paper. 

Note that the directory structure is specified in `decrypt/config.py`. You can change it if you would like.
Most references use this file, but run commands (i.e. `python ...` assume that the directories are unchanged
from the original config.py.

### Datasets and task (Section 3)
- The splits are replicated as above using the load methods
- The task is replicated in the following sections
- We provide code to replicate metric analysis. See the implementation in jupyter notebooks below

To run the notebooks, you should start your jupyter server from the top level `decrypt` directory.
The notebooks have been run using pycharm open from the top level `decrypt` directory.
If you experience import errors it is likely because you are not running from the top level.

### Baselines (Section 4)
Notebook to replicate the four baselines are in `baselines` directory.
Note that a patch will need to be applied to work with the deits solver.


### Curriculum Learning (Section 5) 
See `experiments/curricular.ipynb`

### Model Analysis
See `experiments/model_analysis`


### Misc
Note that details of training and evaluating the models are available in the relevant jupyter
notebooks.

[comment]: <> ([comment]: <> TODO  &#40;## Pre-trained Models&#41;)

[comment]: <> (You can download pretrained models here:)

[comment]: <> (- [My awesome model]&#40;https://drive.google.com/mymodel.pth&#41; trained on ImageNet using parameters x,y,z. )

[comment]: <> (>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained &#40;if applicable&#41;.  Alternatively you can have an additional column in your results table with a link to the models.)

[comment]: <> (## Results)

[comment]: <> (Our model achieves the following performance on :)

[comment]: <> (### [Image Classification on ImageNet]&#40;https://paperswithcode.com/sota/image-classification-on-imagenet&#41;)

[comment]: <> (| Model name         | Top 1 Accuracy  | Top 5 Accuracy |)

[comment]: <> (| ------------------ |---------------- | -------------- |)

[comment]: <> (| My awesome model   |     85%         |      95%       |)

[comment]: <> (>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. )


[comment]: <> (## Contributing)

[comment]: <> (>📋  Pick a licence and describe how to contribute to your code repository. )


