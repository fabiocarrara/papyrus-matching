# Papyrus Fragment Matching with Deep Learning

## Getting Started

```bash
pip install git+https://github.com/fabiocarrara/papyrus-matching.git
```

## Usage
```python
from skiamge.io import imread
from papyrus_matching import FragmentMatcher

# fragments must be RGBA, i.e., np.array with shape (H,W,4)
fragL = imread('left_fragment.png')
fragR = imread('right_fragment.png')

matcher = FragmentMatcher()
posL, posR, scored_displacements, scoresLR = matcher.match(fragL, fragR)
```

Outputs:
 - `posL`, `posR`: (top, left) positions of patches analyzed in the left and right fragment.
 - `scored_displacements`: a (`int` -> `float`) dict mapping a vertical displacement between the two fragments to its score. E.g., `scored_displacements[4]` gives the matching score of the two fragments when displaced vertically by `4 * matcher.stride`. Positive displacements means the right fragment is moved up w.r.t. the left fragment.
 - `scoresLR`: a (`len(posL)`, `len(posR)`)-shaped matrix containing all the matching scores between left and right patches.