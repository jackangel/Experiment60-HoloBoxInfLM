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
NUM_LAYERS = 2      
SEQ_LEN = 256       
ARCHIVE_SIZE = 4096  
SNAPSHOT_RATE = 128 
BATCH_SIZE = 32    
LEARNING_RATE = 5e-4 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = 'infinite_holo_hybrid_v3.pth'
TOKENIZER_FILE = 'hybrid_tokenizer.json'

# --- 1. Stable Box Logic ---
@torch.jit.script
def stable_box_score(q_min, q_max, v_min, v_max):
    inter_min = torch.max(q_min, v_min)
    inter_max = torch.min(q_max, v_max)
    width = F.softplus(inter_max - inter_min)
    log_vol = torch.mean(torch.log(width + 1e-6), dim=-1) 
    return log_vol 

# --- 2. JIT Associative Core (DIMENSION FIXED) ---
@torch.jit.script
def holo_hybrid_scan(x, memory, rarity, 
                    w_k, b_k, w_q, b_q, w_v, b_v, w_out, b_out, 
                    w_gw, b_gw, w_gf, b_gf):
    outputs: list[torch.Tensor] = []
    k_all = F.normalize(F.linear(x, w_k, b_k), p=2.0, dim=-1)
    q_all = F.normalize(F.linear(x, w_q, b_q), p=2.0, dim=-1)
    v_all = torch.tanh(F.linear(x, w_v, b_v))
    
    # Gates are (B, T, 1)
    gw_all = torch.sigmoid(F.linear(x, w_gw, b_gw)) * (0.5 + 0.5 * rarity)
    gf_all = torch.sigmoid(F.linear(x, w_gf, b_gf)) * (0.9 + 0.1 * (1.0 - rarity))
    
    for t in range(x.size(1)):
        k, q, v = k_all[:, t], q_all[:, t], v_all[:, t]
        # These are (B, 1, 1) - Correct for 3D broadcasting
        beta, decay = gw_all[:, t].unsqueeze(-1), gf_all[:, t].unsqueeze(-1)
        
        # Readout: (B, Matrix_Dim, 1) -> (B, Matrix_Dim)
        readout = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1)
        
        # Association: (B, Matrix_Dim, Matrix_Dim)
        association = torch.bmm(v.unsqueeze(-1), k.unsqueeze(1))
        
        # UPDATE: (B, 64, 64) remains 3D. 
        # (NO extra unsqueeze here, which was causing the error)
        memory = (decay * memory) + (beta * association)
        outputs.append(readout)
        
    return F.linear(torch.stack(outputs, dim=1), w_out, b_out), memory

# --- 3. Memory Archive ---
class HoloArchive:
    def __init__(self, size, matrix_dim, embed_dim):
        self.size = size
        self.keys_min = torch.zeros(size, embed_dim).to(DEVICE)
        self.keys_max = torch.zeros(size, embed_dim).to(DEVICE)
        self.values = torch.zeros(size, matrix_dim, matrix_dim).to(DEVICE)
        self.ptr = 0
        self.count = 0

    def add(self, b_min, b_max, matrix):
        self.keys_min[self.ptr] = b_min.mean(dim=(0, 1)).detach()
        self.keys_max[self.ptr] = b_max.mean(dim=(0, 1)).detach()
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
        self.layers = nn.ModuleList([HoloHybridBlock(embed_dim, matrix_dim) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.logit_scale = nn.Parameter(torch.ones(1) * 10.0) 
        
    def forward(self, idx, states=None, archive=None, rarity=None):
        b_min, b_max = self.box_emb(idx)
        x = (b_min + b_max) / 2.0 
        if rarity is None: rarity = torch.ones(x.size(0), x.size(1), 1, device=x.device)
        
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
    def __init__(self, embed_dim, matrix_dim):
        super().__init__()
        self.matrix_dim = matrix_dim
        self.proj_k = nn.Linear(embed_dim, matrix_dim)
        self.proj_q = nn.Linear(embed_dim, matrix_dim)
        self.proj_v = nn.Linear(embed_dim, matrix_dim)
        self.proj_out = nn.Linear(matrix_dim, embed_dim)
        self.gate_write = nn.Linear(embed_dim, 1)
        self.gate_forget = nn.Linear(embed_dim, 1)
        self.ln1, self.ln2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, 4*embed_dim), nn.GELU(), nn.Linear(4*embed_dim, embed_dim))

    def forward(self, x, b_min, b_max, memory=None, archive=None, rarity=None):
        B, T, C = x.shape
        if memory is None: memory = torch.zeros(B, self.matrix_dim, self.matrix_dim, device=x.device)
        
        archive_context = torch.zeros_like(x)
        if archive is not None and archive.count > 10:
            q_min, q_max = b_min.mean(dim=1, keepdim=True), b_max.mean(dim=1, keepdim=True)
            scores = stable_box_score(q_min, q_max, archive.keys_min[:archive.count].unsqueeze(0), archive.keys_max[:archive.count].unsqueeze(0))
            weights = F.softmax(scores, dim=-1).unsqueeze(-1).unsqueeze(-1)
            retrieved_mem = torch.sum(weights * archive.values[:archive.count].unsqueeze(0), dim=1)
            q_vec = F.normalize(self.proj_q(x), p=2.0, dim=-1)
            archive_read = torch.bmm(q_vec, retrieved_mem).squeeze(-1)
            archive_context = self.proj_out(archive_read)

        x_n = self.ln1(x)
        m_out, new_mem = holo_hybrid_scan(x_n, memory, rarity, self.proj_k.weight, self.proj_k.bias, self.proj_q.weight, self.proj_q.bias, self.proj_v.weight, self.proj_v.bias, self.proj_out.weight, self.proj_out.bias, self.gate_write.weight, self.gate_write.bias, self.gate_forget.weight, self.gate_forget.bias)
        
        x = x + m_out + 0.05 * archive_context 
        x = x + self.ffn(self.ln2(x))
        return x, new_mem

# --- 5. Utilities ---
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
    rarity_lookup = torch.tensor([1.0/math.log(counts[i]+2) for i in range(VOCAB_SIZE)]).to(DEVICE)
    
    model = InfiniteHoloGPT(VOCAB_SIZE, EMBED_DIM, MATRIX_DIM, NUM_LAYERS).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    archive = HoloArchive(ARCHIVE_SIZE, MATRIX_DIM, EMBED_DIM)
    
    pbar = tqdm(range(5001), desc="Training Infinite Holo v3.1")
    for step in pbar:
        ix = torch.randint(len(data) - SEQ_LEN - 1, (BATCH_SIZE,))
        xb = torch.stack([data[i:i+SEQ_LEN] for i in ix]).to(DEVICE)
        yb = torch.stack([data[i+1:i+SEQ_LEN+1] for i in ix]).to(DEVICE)
        
        logits, states = model(xb, archive=archive, rarity=rarity_lookup[xb].unsqueeze(-1))
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1))
        
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        
        if step % SNAPSHOT_RATE == 0:
            with torch.no_grad():
                b_min, b_max = model.box_emb(xb); archive.add(b_min, b_max, states[0])
        
        if step > 0 and step % 1000 == 0:
            generate_sample(model, tokenizer, archive)
        if step % 20 == 0: pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    torch.save(model.state_dict(), CHECKPOINT_PATH)

def chat():
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE); tokenizer.decoder = ByteLevelDecoder()
    model = InfiniteHoloGPT(VOCAB_SIZE, EMBED_DIM, MATRIX_DIM, NUM_LAYERS).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    model.eval(); archive = HoloArchive(ARCHIVE_SIZE, MATRIX_DIM, EMBED_DIM)
    states = None
    
    print("\n--- Infinite Holo-Hybrid v3.1 Ready ---")
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
                    b_min, b_max = model.box_emb(idx); archive.add(b_min, b_max, states[0])
        print()

if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_PATH): train()
    chat()