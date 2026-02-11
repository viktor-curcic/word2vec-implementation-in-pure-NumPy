import numpy as np
import matplotlib.pyplot as plt
from data_loader import TinyStoriesLoader
from vocab import Vocabulary
from negative_sampler import NegativeSampler
from model import Word2Vec
from train import Trainer
import re

def main():
    loader = TinyStoriesLoader()
    text = loader.load_text()  
    
    tokens = re.findall(r'\b\w+\b', text)
    stop_words = {"the", "a", "and", "to", "was", "were", "is", "it"}
    tokens = [t for t in tokens if t not in stop_words]
    
    vocab = Vocabulary(tokens, min_count=2, subsample_threshold=1e-4)
    
    model = Word2Vec(vocab.vocab_size, embedding_dim=100, lr=0.025)
    sampler = NegativeSampler(vocab.word_freqs, num_samples=10)
    trainer = Trainer(model, vocab, sampler, window_size=6)
    
    losses = trainer.train_epoch(tokens, num_epochs=10)
    
    trainer.save_model("word2vec_embeddings.pkl")
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss - Word2Vec on TinyStories')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png', dpi=100, bbox_inches='tight')
    
    evaluate_similarities(model, vocab)
    
    return model, vocab

def evaluate_similarities(model, vocab):
    embeddings = model.W_in + model.W_out
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.clip(norms, 1e-9, None)
    
    test_words = ['dog', 'cat', 'run', 'happy', 'big', 'small', 'house', 'tree']
    
    print("Word Similarity Evaluation")
    
    for word in test_words:
        if word in vocab.word_to_idx:
            idx = vocab.word_to_idx[word]
            word_vec = normalized[idx]
            
            similarities = np.dot(normalized, word_vec)
            
            top_indices = np.argsort(similarities)[-6:-1][::-1]
            
            print(f"\nWords similar to '{word}':")
            for i in top_indices:
                similarity = similarities[i]
                print(f"  {vocab.idx_to_word[i]:15s} similarity: {similarity:.3f}")
        else:
            print(f"\nWord '{word}' not in vocabulary")

if __name__ == "__main__":
    model, vocab = main()