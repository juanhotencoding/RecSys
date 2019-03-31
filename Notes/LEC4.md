## Recommender Systems Item kNN

### Neighborhood based models

### User-based

### Neighborhood methods
key idea
- we extrapolate the values of missing ratings, $\hat{r}_{uj}$

### How many neighbors
Large k
- means that each prediction is closer to a global mean
- less personalization

Small k
- sensitive to neighbor noise

Interestingly
- if you had an oracle to tell you the exact right k for each user
- you can be very accurate
- no way to know that in advance

Similarity threshold
- may want to combine k value with minimum similarity threshold
- otherwise, when users have few close neighbors
  - distant users get pulled in to get to k
- Zero is a common threshold
  - don't profile with negative similarity
- Might be hard to find k neighbors that are close
  - "black sheep" users