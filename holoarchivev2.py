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
import random

# --- Hyperparameters ---
VOCAB_SIZE = 4096    
EMBED_DIM = 512    
MATRIX_DIM = 64     
NUM_HEADS = 16       
HEAD_DIM = MATRIX_DIM // NUM_HEADS # 4
NUM_LAYERS = 2      
SEQ_LEN = 256       
ARCHIVE_SIZE = 4096  
SNAPSHOT_RATE = 128 
BATCH_SIZE = 32    
LEARNING_RATE = 5e-4 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = 'infinite_holo_hybrid_v3_optimized.pth'
TOKENIZER_FILE = 'hybrid_tokenizer.json'

# --- 1. Stable Box Logic ---
@torch.jit.script
def stable_box_score(q_min, q_max, v_min, v_max):
    inter_min = torch.max(q_min, v_min)
    inter_max = torch.min(q_max, v_max)
    width = F.softplus(inter_max - inter_min)
    log_vol = torch.mean(torch.log(width + 1e-6), dim=-1) 
    return log_vol 

# --- 2. Parallel Associative Core ---
def holo_parallel_scan(k, q, v, gf, gw):
    B, T, H, D = k.shape
    # Updates: Outer product (v @ k.T) scaled by write gate
    updates = torch.matmul(v.unsqueeze(-1), k.unsqueeze(-2)) * gw.unsqueeze(-1).unsqueeze(-1)
    
    # Log-space accumulation for perfect stability
    log_gf = torch.log(gf.clamp(min=1e-8)).unsqueeze(-1).unsqueeze(-1)
    cum_log_gf = torch.cumsum(log_gf, dim=1)
    exp_cum_gf = torch.exp(cum_log_gf)
    
    # Prefix-sum scan logic
    weighted_updates = updates / (exp_cum_gf + 1e-8)
    state = torch.cumsum(weighted_updates, dim=1) * exp_cum_gf
    
    # Readout: M @ q
    readout = torch.matmul(state, q.unsqueeze(-1)).squeeze(-1)
    return readout, state[:, -1]

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
        self.logit_scale = nn.Parameter(torch.ones(1) * 12.0)
        
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
        
        nn.init.constant_(self.gate_forget.bias, 12.0)
        nn.init.constant_(self.gate_write.bias, -12.0)
        
        self.archive_gate = nn.Linear(embed_dim, 1)
        nn.init.constant_(self.archive_gate.bias, -4.0) 
        
        self.ln1, self.ln2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, 4*embed_dim), nn.GELU(), nn.Linear(4*embed_dim, embed_dim))

    def forward(self, x, b_min, b_max, memory=None, archive=None, rarity=None):
        B, T, C = x.shape
        
        # --- Archive Retrieval Logic (Multi-Head Corrected) ---
        archive_context = torch.zeros_like(x)
        if archive is not None and archive.count > 10:
            q_min, q_max = b_min.mean(dim=1, keepdim=True), b_max.mean(dim=1, keepdim=True)
            valid_count = archive.count
            scores = stable_box_score(q_min, q_max, archive.keys_min[:valid_count].unsqueeze(0), archive.keys_max[:valid_count].unsqueeze(0))
            weights = F.softmax(scores, dim=-1).reshape(B, 1, valid_count, 1, 1, 1)
            
            # Weighted average of archive matrices
            fused_mem = torch.sum(weights * archive.values[:valid_count].reshape(1, 1, valid_count, self.num_heads, self.head_dim, self.head_dim), dim=2)
            q_ret = F.normalize(self.proj_q(x).reshape(B, T, self.num_heads, self.head_dim), p=2.0, dim=-1)
            archive_read = torch.matmul(fused_mem, q_ret.unsqueeze(-1)).squeeze(-1)
            archive_context = self.proj_out(archive_read.reshape(B, T, -1))

        x_n = self.ln1(x)
        # Projections
        k = F.normalize(self.proj_k(x_n).reshape(B, T, self.num_heads, self.head_dim), p=2.0, dim=-1)
        q = F.normalize(self.proj_q(x_n).reshape(B, T, self.num_heads, self.head_dim), p=2.0, dim=-1)
        v = torch.tanh(self.proj_v(x_n).reshape(B, T, self.num_heads, self.head_dim))
        
        # Optimization Gating
        gf = 1.0 - torch.pow(1.0 - torch.sigmoid(self.gate_forget(x_n)), 2).reshape(B, T, self.num_heads)
        gw = torch.pow(torch.sigmoid(self.gate_write(x_n)), 2).reshape(B, T, self.num_heads)
        
        gw = gw * (0.5 + 0.5 * rarity.reshape(B, T, 1))
        gf = gf * (0.9 + 0.1 * (1.0 - rarity.reshape(B, T, 1)))

        if T > 1 or memory is None:
            if memory is None: memory = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device)
            m_out_heads, next_mem = holo_parallel_scan(k, q, v, gf, gw)
            m_out = self.proj_out(m_out_heads.reshape(B, T, -1))
        else:
            # RECURRENT INFERENCE PATH (Fixed Matmuls)
            k_s, q_s, v_s = k.squeeze(1), q.squeeze(1), v.squeeze(1) # (B, H, D)
            gf_s, gw_s = gf.squeeze(1), gw.squeeze(1) # (B, H)
            
            # Association: (B, H, D, 1) @ (B, H, 1, D) -> (B, H, D, D)
            assoc = torch.matmul(v_s.unsqueeze(-1), k_s.unsqueeze(-2))
            next_mem = (gf_s.reshape(B, self.num_heads, 1, 1) * memory) + (gw_s.reshape(B, self.num_heads, 1, 1) * assoc)
            
            # Readout: (B, H, D, D) @ (B, H, D, 1) -> (B, H, D)
            m_out_heads = torch.matmul(next_mem, q_s.unsqueeze(-1)).squeeze(-1)
            m_out = self.proj_out(m_out_heads.reshape(B, 1, -1))

        # Add Context and FFN
        g_mem = torch.sigmoid(self.archive_gate(x))
        x = x + m_out + (g_mem * archive_context)
        x = x + self.ffn(self.ln2(x))
        return x, next_mem

# --- 5. Training / Utils ---
def generate_sample(model, tokenizer, archive, length=50):
    model.eval()
    context = torch.tensor(tokenizer.encode("ROMEO:").ids).unsqueeze(0).to(DEVICE)
    states = None
    output = "VALIDATION: "
    with torch.no_grad():
        for _ in range(length):
            logits, states = model(context, states=states, archive=archive)
            next_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            output += tokenizer.decode([next_id.item()])
            context = next_id
    print(f"\n{output}\n")
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
    
    pbar = tqdm(range(5001), desc="Training Optimized Holo V3")
    for step in pbar:
        curr_len = random.randint(SEQ_LEN // 2, SEQ_LEN)
        ix = torch.randint(len(data) - curr_len - 1, (BATCH_SIZE,))
        xb = torch.stack([data[i:i+curr_len] for i in ix]).to(DEVICE)
        yb = torch.stack([data[i+1:i+curr_len+1] for i in ix]).to(DEVICE)
        
        batch_rarity = rarity_lookup[xb].unsqueeze(-1)
        logits, states = model(xb, archive=archive, rarity=batch_rarity)
        
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), yb.reshape(-1))
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
    model.eval(); archive = HoloArchive(ARCHIVE_SIZE, NUM_HEADS, HEAD_DIM, EMBED_DIM)
    states = None
    
    print("\n--- Infinite Holo-Hybrid Optimized Ready ---")
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