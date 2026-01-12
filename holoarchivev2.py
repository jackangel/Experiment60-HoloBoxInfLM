import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import requests
import os
import math
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder 
from collections import Counter

# --- Hyperparameters ---
VOCAB_SIZE = 4096    
EMBED_DIM = 512    
MATRIX_DIM = 64     
NUM_HEADS = 16       
HEAD_DIM = MATRIX_DIM // NUM_HEADS
NUM_LAYERS = 2      
SEQ_LEN = 256       
ARCHIVE_SIZE = 4096  
SNAPSHOT_RATE = 128 
BATCH_SIZE = 32    
LEARNING_RATE = 5e-4 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = 'infinite_holo_hybrid_v3_enhanced.pth'
TOKENIZER_FILE = 'hybrid_tokenizer.json'

# --- 1. Stable Box Logic ---
@torch.jit.script
def stable_box_score(q_min, q_max, v_min, v_max):
    inter_min = torch.max(q_min, v_min)
    inter_max = torch.min(q_max, v_max)
    width = F.softplus(inter_max - inter_min)
    log_vol = torch.mean(torch.log(width + 1e-6), dim=-1) 
    return log_vol 

# --- 2. JIT Associative Core (MULTI-HEAD FIX) ---
@torch.jit.script
def holo_hybrid_scan(x, memory, rarity, 
                    w_k, b_k, w_q, b_q, w_v, b_v, w_out, b_out, 
                    w_gw, b_gw, w_gf, b_gf,
                    num_heads: int, head_dim: int):
    # Inputs:
    # x: (B, T, Embed)
    # memory: (B, Heads, Head_Dim, Head_Dim)
    
    B, T, C = x.size()
    outputs: list[torch.Tensor] = []
    
    # Project and reshape for Multi-Head: (B, T, Heads, Head_Dim)
    k_all = F.normalize(F.linear(x, w_k, b_k).view(B, T, num_heads, head_dim), p=2.0, dim=-1)
    q_all = F.normalize(F.linear(x, w_q, b_q).view(B, T, num_heads, head_dim), p=2.0, dim=-1)
    v_all = torch.tanh(F.linear(x, w_v, b_v).view(B, T, num_heads, head_dim))
    
    # Gates: (B, T, Heads)
    gw_all = torch.sigmoid(F.linear(x, w_gw, b_gw).view(B, T, num_heads)) * (0.5 + 0.5 * rarity)
    gf_all = torch.sigmoid(F.linear(x, w_gf, b_gf).view(B, T, num_heads)) * (0.9 + 0.1 * (1.0 - rarity))
    
    for t in range(T):
        k, q, v = k_all[:, t], q_all[:, t], v_all[:, t]
        
        # Gates: (B, Heads, 1, 1) for broadcasting
        beta = gw_all[:, t].unsqueeze(-1).unsqueeze(-1)
        decay = gf_all[:, t].unsqueeze(-1).unsqueeze(-1)
        
        # Readout: (B, Heads, Dim, Dim) @ (B, Heads, Dim, 1) -> (B, Heads, Dim)
        readout = torch.matmul(memory, q.unsqueeze(-1)).squeeze(-1)
        
        # Association: 
        # v: (B, Heads, Dim) -> (B, Heads, Dim, 1)
        # k: (B, Heads, Dim) -> (B, Heads, 1, Dim) (Use -2 to insert before last dim)
        association = torch.matmul(v.unsqueeze(-1), k.unsqueeze(-2))
        
        # Update Memory
        memory = (decay * memory) + (beta * association)
        outputs.append(readout)
    
    # Stack: (B, T, Heads, Head_Dim)
    stacked = torch.stack(outputs, dim=1)
    # Flatten heads: (B, T, Heads * Head_Dim)
    stacked = stacked.view(B, T, num_heads * head_dim)
    
    return F.linear(stacked, w_out, b_out), memory

# --- 3. Memory Archive ---
class HoloArchive:
    def __init__(self, size, num_heads, head_dim, embed_dim):
        self.size = size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.keys_min = torch.zeros(size, embed_dim).to(DEVICE)
        self.keys_max = torch.zeros(size, embed_dim).to(DEVICE)
        self.values = torch.zeros(size, num_heads, head_dim, head_dim).to(DEVICE)
        self.ptr = 0
        self.count = 0

    def add(self, b_min, b_max, matrix):
        # Store LAST token of batch as key
        self.keys_min[self.ptr] = b_min[:, -1, :].mean(dim=0).detach()
        self.keys_max[self.ptr] = b_max[:, -1, :].mean(dim=0).detach()
        self.values[self.ptr] = matrix.mean(dim=0).detach()
        self.ptr = (self.ptr + 1) % self.size
        self.count = min(self.count + 1, self.size)

# --- 4. Enhanced Hybrid Architecture ---
class BoxEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center = nn.Embedding(vocab_size, embed_dim)
        self.offset = nn.Embedding(vocab_size, embed_dim)
        nn.init.xavier_uniform_(self.center.weight)
        nn.init.constant_(self.offset.weight, -2.0)
    def forward(self, idx):
        c, o = self.center(idx), F.softplus(self.offset(idx))
        return c - o, c + o

class InfiniteHoloGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, matrix_dim, layers):
        super().__init__()
        self.box_emb = BoxEmbedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            HoloHybridBlock(embed_dim, matrix_dim, NUM_HEADS) for _ in range(layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.logit_scale = nn.Parameter(torch.ones(1) * 10.0) 
        
    def forward(self, idx, states=None, archive=None, rarity=None):
        b_min, b_max = self.box_emb(idx)
        x = (b_min + b_max) / 2.0 
        
        if rarity is None: 
            rarity = torch.ones(x.size(0), x.size(1), 1, device=x.device)
        
        new_states = []
        if states is None: states = [None] * len(self.layers)
        
        for i, layer in enumerate(self.layers):
            x, s = layer(x, b_min, b_max, states[i], archive, rarity)
            new_states.append(s)
            
        x = self.ln_f(x)
        logits = F.linear(x, self.box_emb.center.weight) 
        logits = logits * (self.logit_scale / (EMBED_DIM**0.5))
        return logits, new_states

class HoloHybridBlock(nn.Module):
    def __init__(self, embed_dim, matrix_dim, num_heads):
        super().__init__()
        self.matrix_dim = matrix_dim
        self.num_heads = num_heads
        self.head_dim = matrix_dim // num_heads
        
        self.proj_k = nn.Linear(embed_dim, matrix_dim)
        self.proj_q = nn.Linear(embed_dim, matrix_dim)
        self.proj_v = nn.Linear(embed_dim, matrix_dim)
        self.proj_out = nn.Linear(matrix_dim, embed_dim)
        
        self.gate_write = nn.Linear(embed_dim, num_heads)
        self.gate_forget = nn.Linear(embed_dim, num_heads)
        
        # Adaptive Memory Gate
        self.archive_gate = nn.Linear(embed_dim, 1)
        nn.init.constant_(self.archive_gate.bias, -3.0) 
        
        self.ln1, self.ln2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, 4*embed_dim), nn.GELU(), nn.Linear(4*embed_dim, embed_dim))

    def forward(self, x, b_min, b_max, memory=None, archive=None, rarity=None):
        B, T, C = x.shape
        if memory is None: 
            memory = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device)
        
        archive_context = torch.zeros_like(x)
        
        # --- Top-K Sparse Retrieval ---
        if archive is not None and archive.count > 10:
            q_min, q_max = b_min.mean(dim=1, keepdim=True), b_max.mean(dim=1, keepdim=True)
            
            valid_count = archive.count
            keys_min = archive.keys_min[:valid_count].unsqueeze(0)
            keys_max = archive.keys_max[:valid_count].unsqueeze(0)
            
            scores = stable_box_score(q_min, q_max, keys_min, keys_max) # (B, 1, Count)
            
            k = min(8, valid_count)
            top_scores, top_indices = torch.topk(scores, k, dim=-1) # (B, 1, K)
            
            weights = F.softmax(top_scores, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
            flat_indices = top_indices.squeeze(1).view(-1)
            retrieved_raw = archive.values[flat_indices]
            retrieved_raw = retrieved_raw.view(B, k, self.num_heads, self.head_dim, self.head_dim)
            
            fused_mem = torch.sum(weights.squeeze(1) * retrieved_raw, dim=1) 
            
            q_curr = F.normalize(self.proj_q(x).view(B, T, self.num_heads, self.head_dim), p=2.0, dim=-1)
            
            fused_mem_expanded = fused_mem.unsqueeze(1) 
            archive_read = torch.matmul(fused_mem_expanded, q_curr.unsqueeze(-1)).squeeze(-1)
            
            archive_read = archive_read.view(B, T, -1)
            archive_context = self.proj_out(archive_read)

        x_n = self.ln1(x)
        m_out, new_mem = holo_hybrid_scan(
            x_n, memory, rarity, 
            self.proj_k.weight, self.proj_k.bias, 
            self.proj_q.weight, self.proj_q.bias, 
            self.proj_v.weight, self.proj_v.bias, 
            self.proj_out.weight, self.proj_out.bias, 
            self.gate_write.weight, self.gate_write.bias, 
            self.gate_forget.weight, self.gate_forget.bias,
            self.num_heads, self.head_dim
        )
        
        if archive is not None and archive.count > 10:
            g_mem = torch.sigmoid(self.archive_gate(x)) 
            x = x + m_out + (g_mem * archive_context)
        else:
            x = x + m_out
            
        x = x + self.ffn(self.ln2(x))
        return x, new_mem

# --- 5. Utilities ---
def generate_sample(model, tokenizer, archive, length=50):
    model.eval()
    context = torch.tensor(tokenizer.encode("ROMEO:").ids).unsqueeze(0).to(DEVICE)
    states = None
    output = "VALIDATION: "
    print(f"\n--- Generating (Archive Size: {archive.count}) ---")
    with torch.no_grad():
        for _ in range(length):
            logits, states = model(context, states=states, archive=archive)
            next_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            output += tokenizer.decode([next_id.item()])
            context = next_id
    print(f"{output}\n")
    model.train()

def train():
    if not os.path.exists('input.txt'):
        r = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
        with open('input.txt', 'w') as f: f.write(r.text)
    
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(); tokenizer.decoder = ByteLevelDecoder()
    tokenizer.train(["input.txt"], BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["[UNK]", "[PAD]"]))
    tokenizer.save(TOKENIZER_FILE)
    
    with open('input.txt', 'r') as f: text = f.read()
    data = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)
    counts = Counter(data.tolist())
    rarity_lookup = torch.zeros(VOCAB_SIZE).to(DEVICE)
    for i in range(VOCAB_SIZE):
        c = counts[i]
        if c > 0: rarity_lookup[i] = 1.0 / math.log(c + 2)
    
    model = InfiniteHoloGPT(VOCAB_SIZE, EMBED_DIM, MATRIX_DIM, NUM_LAYERS).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    archive = HoloArchive(ARCHIVE_SIZE, NUM_HEADS, HEAD_DIM, EMBED_DIM)
    
    pbar = tqdm(range(5001), desc="Training Infinite Holo v3.1 Enhanced")
    for step in pbar:
        ix = torch.randint(len(data) - SEQ_LEN - 1, (BATCH_SIZE,))
        xb = torch.stack([data[i:i+SEQ_LEN] for i in ix]).to(DEVICE)
        yb = torch.stack([data[i+1:i+SEQ_LEN+1] for i in ix]).to(DEVICE)
        
        batch_rarity = rarity_lookup[xb].unsqueeze(-1)
        
        logits, states = model(xb, archive=archive, rarity=batch_rarity)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1))
        
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        
        if step % SNAPSHOT_RATE == 0:
            with torch.no_grad():
                b_min, b_max = model.box_emb(xb)
                archive.add(b_min, b_max, states[0])
        
        if step > 0 and step % 1000 == 0:
            generate_sample(model, tokenizer, archive)
        if step % 20 == 0: pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    torch.save(model.state_dict(), CHECKPOINT_PATH)

def chat():
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE); tokenizer.decoder = ByteLevelDecoder()
    model = InfiniteHoloGPT(VOCAB_SIZE, EMBED_DIM, MATRIX_DIM, NUM_LAYERS).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    archive = HoloArchive(ARCHIVE_SIZE, NUM_HEADS, HEAD_DIM, EMBED_DIM)
    states = None
    
    print("\n--- Infinite Holo-Hybrid v3.1 Enhanced Ready ---")
    while True:
        prompt = input("\nYou: ")
        if not prompt or prompt.lower() == 'exit': break
        idx = torch.tensor(tokenizer.encode(prompt).ids).unsqueeze(0).to(DEVICE)
        print("Bot: ", end="", flush=True)
        with torch.no_grad():
            for _ in range(150):
                logits, states = model(idx, states=states, archive=archive)
                probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
                next_id = torch.multinomial(probs, 1)
                word = tokenizer.decode([next_id.item()]); print(word, end="", flush=True)
                idx = next_id
                if _ % SNAPSHOT_RATE == 0:
                    b_min, b_max = model.box_emb(idx)
                    archive.add(b_min, b_max, states[0])
        print()

if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_PATH): train()
    chat()