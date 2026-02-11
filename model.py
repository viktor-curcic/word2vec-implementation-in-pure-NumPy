import numpy as np

class Word2Vec:
    
    def __init__(self, vocab_size, embedding_dim=100, lr=0.025):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = lr
        
        limit = np.sqrt(6 / (vocab_size + embedding_dim))
        self.W_in = np.random.uniform(-limit, limit, (vocab_size, embedding_dim))
        self.W_out = np.random.uniform(-limit, limit, (vocab_size, embedding_dim))
        
    def forward(self, center_idx, context_idx, negative_idxs):
        v_center = self.W_in[center_idx]
        v_context = self.W_out[context_idx]

        pos_score = np.dot(v_center, v_context)
        pos_prob = self._sigmoid(pos_score)

        neg_vectors = self.W_out[negative_idxs]
        neg_scores = np.dot(neg_vectors, v_center)
        neg_probs = self._sigmoid(neg_scores)

        loss = -np.log(pos_prob + 1e-10) - np.sum(np.log(1 - neg_probs + 1e-10))

        self.cache = {
            'center_idx': center_idx,
            'context_idx': context_idx,
            'negative_idxs': negative_idxs,
            'v_center': v_center,
            'v_context': v_context,
            'neg_vectors': neg_vectors,
            'pos_prob': pos_prob,
            'neg_probs': neg_probs
        }

        return loss


    def backward(self):
        c = self.cache

        grad_pos = c['pos_prob'] - 1
        grad_center = grad_pos * c['v_context']

        for i, neg_idx in enumerate(c['negative_idxs']):
            grad_neg = c['neg_probs'][i]
            grad_center += grad_neg * c['neg_vectors'][i]
            self.W_out[neg_idx] -= self.lr * (grad_neg * c['v_center'])

        self.W_in[c['center_idx']] -= self.lr * grad_center
        self.W_out[c['context_idx']] -= self.lr * (grad_pos * c['v_center'])

    
    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -10, 10)  
        return 1 / (1 + np.exp(-x))
    
    def get_embeddings(self):
        return self.W_in