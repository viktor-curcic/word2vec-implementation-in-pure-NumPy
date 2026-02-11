import numpy as np
from tqdm import tqdm
import pickle

class Trainer:
    
    def __init__(self, model, vocab, sampler, window_size=5):
        self.model = model
        self.vocab = vocab
        self.sampler = sampler
        self.window_size = window_size
        
    def generate_training_pairs(self, tokens, batch_size=100):

        pairs = []
        
        for i, center_word in enumerate(tokens):
            center_idx = self.vocab.word_to_idx.get(center_word)
            if center_idx is None:
                continue
                
            window = np.random.randint(1, self.window_size + 1)
            start = max(0, i - window)
            end = min(len(tokens), i + window + 1)
            
            for j in range(start, end):
                if i == j:
                    continue
                    
                context_word = tokens[j]
                context_idx = self.vocab.word_to_idx.get(context_word)
                if context_idx is None:
                    continue
                
                if not self.vocab.subsample_word(center_word):
                    continue

                if not self.vocab.subsample_word(context_word):
                    continue
                    
                pairs.append((center_idx, context_idx))
                
                if len(pairs) >= batch_size:
                    yield pairs
                    pairs = []
        
        if pairs:
            yield pairs
    
    def train_epoch(self, tokens, num_epochs=10):
        losses = []

        initial_lr = self.model.lr
        min_lr = initial_lr * 0.05   

        for epoch in range(num_epochs):
            epoch_loss = 0
            num_pairs = 0

            lr = max(initial_lr * (0.9 ** epoch), min_lr)
            self.model.lr = lr

            pbar = tqdm(self.generate_training_pairs(tokens), desc=f"Epoch {epoch+1}")

            for batch in pbar:
                batch_loss = 0

                for center_idx, context_idx in batch:
                    negative_idxs = self.sampler.sample(center_idx, context_idx)

                    loss = self.model.forward(center_idx, context_idx, negative_idxs)
                    self.model.backward()

                    batch_loss += loss
                    num_pairs += 1

                avg_batch_loss = batch_loss / len(batch) if batch else 0
                epoch_loss += batch_loss
                pbar.set_postfix({'loss': f'{avg_batch_loss:.3f}', 'lr': f'{self.model.lr:.5f}'})

            avg_epoch_loss = epoch_loss / num_pairs if num_pairs > 0 else 0
            losses.append(avg_epoch_loss)

            print(f"Epoch {epoch+1} completed. Avg loss: {avg_epoch_loss:.3f}")

        return losses


    
    def save_model(self, path):
        embeddings = {
            'embeddings': self.model.get_embeddings(),
            'idx_to_word': self.vocab.idx_to_word,
            'word_to_idx': self.vocab.word_to_idx
        }
        with open(path, 'wb') as f:
            pickle.dump(embeddings, f)