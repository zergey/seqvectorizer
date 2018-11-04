# seqvectorizer 0.0.1
Turn sequence of any tokens into a fix-length representation vector.
Mostly suitable for short texts or event sequences.

## Install
clone the repo, then install:
```
git clone https://github.com/zergey/seqvectorizer
python setup.py install
```


## Usage

Simple transformation:
```python
import re
from sklearn.datasets import fetch_20newsgroups

from seqvectorizer import SeqVectorizer


token_pattern=r"(?u)\b\w\w+\b"
token_pattern = re.compile(token_pattern)
str2words = lambda doc: token_pattern.findall(doc.lower())

newsgroups_train = fetch_20newsgroups(subset='train')
vec_train = [str2words(x) for x in newsgroups_train["data"][:100]]
newsgroups_test = fetch_20newsgroups(subset='train')
vec_test = [str2words(x) for x in newsgroups_test["data"][:1]]

sv = SeqVectorizer(max_seq_len=50, verbose=2, max_iter=10)
sv.fit(vec_train)

print(sv.score())
print(sv.transform(vec_test))
```
