import torch
from transformers import BertForMaskedLM, BertTokenizer, AutoTokenizer, AutoModel
from typing import List

class BertMLMSynonymReplacer:
    def __init__(self, top_k: int = 15, bert_model_name: str = 'bert-base-uncased'):
        print(f"Initializing BERT-MLM-based synonym replacer ({bert_model_name}).")
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.model = BertForMaskedLM.from_pretrained(bert_model_name)
        self.top_k = top_k
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device).eval()

    def get_candidates(self, sentence: str, word_index: int) -> List[str]:
        words = sentence.split()
        if word_index >= len(words): return []
        original_word = words[word_index]
        words[word_index] = self.tokenizer.mask_token
        masked_sentence = " ".join(words)
        inputs = self.tokenizer(masked_sentence, return_tensors='pt').to(self.device)
        mask_token_index = torch.where(inputs.input_ids == self.tokenizer.mask_token_id)[1]
        
        if mask_token_index.numel() == 0: return []
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits[0, mask_token_index[0], :]
        top_k_tokens = torch.topk(logits, self.top_k, dim=0).indices.tolist()
        candidates = set()
        
        for token_id in top_k_tokens:
            candidate = self.tokenizer.decode(token_id)
            if candidate not in [self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.mask_token] and \
               ' ' not in candidate and candidate.isalpha() and candidate.lower() != original_word.lower():
                candidates.add(candidate)
        return list(candidates)[:self.top_k]

class SemanticsModel:
    def __init__(self, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing semantic model (all-MiniLM-L6-v2) on {self.device}...")
        self.model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def get_embedding(self, text: str):
        import torch.nn.functional as F
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sentence_embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return F.normalize(sentence_embedding, p=2, dim=1)

    def semantic_similarity(self, text1: str, text2: str) -> float:
        if text1.strip() == text2.strip(): return 1.0
        emb1, emb2 = self.get_embedding(text1), self.get_embedding(text2)
        return max(0.0, min(1.0, torch.mm(emb1, emb2.transpose(0, 1)).item()))
