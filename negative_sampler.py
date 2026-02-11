import numpy as np

class NegativeSampler:
    
    def __init__(self, word_freqs, num_samples=5):
        self.num_samples = num_samples
        self.word_freqs = word_freqs
        self.vocab_size = len(word_freqs)
        
        self.cumulative_dist = np.cumsum(word_freqs)
        
    def sample(self, center_idx, context_idx):
        samples = []
        
        for _ in range(self.num_samples):
            r = np.random.random()
            sample_idx = np.searchsorted(self.cumulative_dist, r)
            
            while sample_idx == center_idx or sample_idx == context_idx:
                r = np.random.random()
                sample_idx = np.searchsorted(self.cumulative_dist, r)
                
            samples.append(sample_idx)
            
        return samples