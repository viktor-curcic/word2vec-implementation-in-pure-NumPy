import numpy as np
from collections import Counter
import re

class Vocabulary:
    
    def __init__(self, text, min_count=5, subsample_threshold=1e-5):
        self.min_count = min_count
        self.subsample_threshold = subsample_threshold
        self.build_vocab(text)
        self.precompute_subsampling()
        
    def build_vocab(self, tokens):
        words = [w.lower() for w in tokens]
        word_counts = Counter(words)

        self.word_counts = {w: c for w, c in word_counts.items()
                        if c >= self.min_count}

        self.idx_to_word = list(self.word_counts.keys())
        self.word_to_idx = {w: i for i, w in enumerate(self.idx_to_word)}

        self.vocab_size = len(self.idx_to_word)

        total_words = sum(self.word_counts.values())
        freqs = np.array([self.word_counts[w] for w in self.idx_to_word])
        freqs = freqs / total_words

        self.word_freqs = np.power(freqs, 0.75)
        self.word_freqs /= self.word_freqs.sum()

        
    def precompute_subsampling(self):
        total_words = sum(self.word_counts.values())
        self.discard_probs = {}
        
        for word, count in self.word_counts.items():
            freq = count / total_words
            prob = 1 - np.sqrt(self.subsample_threshold / freq)
            self.discard_probs[word] = max(0, prob)
            
    def subsample_word(self, word):
        return np.random.random() > self.discard_probs.get(word, 0)
