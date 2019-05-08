## Recommender Systems Ratings

### User Profile
- the information about the user that we use to make recommendations

Two types
- explicit: the user tells you their preferences
- implicit: you infer the user's preferences from their actions
  - usually transform them into preferences somehow

Not always straightforward to create profies frmo implicit data. 

Any conversion from raw data to preferences encodes assumptions about what users' actions mean

### Some Notation
User: a unique individual (for the system's purposes), possibly the target of personalization
$u$ or $v$

Item: a unique item, possibly to be recommended to a user
- $i$ or $j$
- $I$ all items

Rating / preferences: a numerical association between some $u$ and $i$ 

Any user, item pair can be a dependent variable
