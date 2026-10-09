# Tokenizer comparison results

Corpus: `eng_fra_songs.txt`.
566 aligned pairs; seed 42; 904 training and 228 held-out sentences.
Both languages of each pair remain in the same split. No model is trained on held-out sentences.
Pairs connected by identical repeated lyrics are grouped before splitting; exact sentence overlap is zero.

Metric: total Unicode characters / total tokens, excluding inserted BOS/EOS. Higher means more compression.
Vocabulary size includes four special tokens and 256 reserved byte tokens in both models.

| Model | Requested vocabulary | Actual vocabulary | Train chars/token | Held-out chars/token |
| --- | ---: | ---: | ---: | ---: |
| bpe | 384 | 384 | 1.5448 | 1.5404 |
| sentencepiece_unigram | 384 | 384 | 1.4812 | 1.4539 |
| bpe | 512 | 512 | 2.0652 | 1.9874 |
| sentencepiece_unigram | 512 | 512 | 1.9075 | 1.8056 |
| bpe | 640 | 640 | 2.3836 | 2.2615 |
| sentencepiece_unigram | 640 | 640 | 2.1386 | 1.9610 |
| bpe | 768 | 768 | 2.6401 | 2.4440 |
| sentencepiece_unigram | 768 | 768 | 2.3250 | 2.0727 |
| bpe | 1024 | 1024 | 3.0843 | 2.7042 |
| sentencepiece_unigram | 1024 | 1024 | 2.6039 | 2.2356 |

All measured corpus round trips succeeded; no unknown tokens were emitted.
These small-corpus results measure compression, not downstream model quality.
The algorithms can produce different segmentations; matching token IDs is not expected.
Unigram selects a probable segmentation from a trained probabilistic vocabulary; BPE follows merge ranks.
