from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer






class LLaMA:

    def __init__(self, model:Transformer, tokenizer:SentencePieceProcessor, args:ModelArgs) -> None:
        
        self.model = model
        self.args = args
        self.tokenizer = tokenizer

    @staticmethod
    def build(chkpt_dir:str, tok_pth:str, load_model:bool, max_seq_len:int, max_batch_size:int, device:str):
        start = time.time()
        if load_model:
            chkpt = sorted(Path(chkpt_dir).glob('*.pth'))
            assert len(chkpt) > 0, "No checkpoints file found"
            chk_pth = chkpt[0]
            checkpoint = torch.load(chk_pth, map_location='cpu')
            print(f'Loaded checkpoint in {time.time() - start:.2f}s')

        with open(Path(chkpt_dir)/ "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len = max_seq_len,
            max_batch_size = max_batch_size,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tok_pth)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)  

        model = Transformer(model_args).to(device)

        if load_model:
            start = time.time()
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f'Loaded model in {time.time() - start:.2f}s')

        return LLaMA(model, tokenizer, model_args)   

    def get_top_p(self, x, top_p):
        # x shape -> (b, vocab_size)
        probs, idxs = torch.sort(x, dim=-1, descending=True)
        cumsum = torch.cumsum(probs, dim=-1)
        mask = cumsum - probs > top_p
        probs[mask] = 0.0

        # redistribute probs so they sum up to 1
        probs.div_(probs.sum(dim=-1, keepdim=True))
        # select 1 token from new distribution
        next_token = torch.multinomial(probs, num_samples=1)
        # map back to orginal index to get the index 
        next_token = torch.gather(probs, -1, next_token)
        return next_token


    def text_completion(self, prompts: list[str], temp: float=0.6, top_p: float=0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1

        inp_tokens = [self.tokenizer.Encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        batch_size = len(prompts)
        assert batch_size <= self.args.max_batch_size, "Batch size too large"
        max_inp_len = max(len(prompt) for prompt in prompts)
        assert max_inp_len <= self.args.max_seq_len, "Prompt too long"
        gen_length = min(self.args.max_seq_len, max_gen_len + max_inp_len)

        pad_tok_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, gen_length), pad_tok_id, dtype=torch.long, device=device) 
        for k,t in enumerate(inp_tokens):
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)   
        
        eos_reached = torch.tensor([False] * batch_size, device=device)
        inp_mask = tokens != pad_tok_id   # False if padding token

        for start_pos in tqdm(range(1, gen_length), desc="Generating tokens...."):
            with torch.no_grad():
                logits = self.model.forward(tokens[:, start_pos-1:start_pos], start_pos)
                if temp > 0:
                    # top_p strategy after scaling logits by temp
                    probs = torch.softmax(logits[:, -1] / temp, dim = -1)
                    new_tokens = self.get_top_p(probs, top_p)
                else:
                    # greedy
                    new_tokens = torch.argmax(torch.softmax(logits[:, -1], dim=-1), dim=-1)

                new_tokens = new_tokens.reshape(-1)
                # replace in tokens if cur_pos is a padding token
                # Mask true hai toh keep og token, false hai toh use new token
                new_tokens = torch.where(inp_mask[:, start_pos], tokens[:, start_pos], new_tokens)
                tokens[:, start_pos] = new_tokens

                # EOS reached if EOS token found AND cur_pos is not a padding token
                eos_reached |= (~inp_mask[:, start_pos]) and (new_tokens == self.tokenizer.eos_id())
                if all(eos_reached):
                    break
                
        
        out_texts = []
        out_tokens = []
        for idx, prompt_tokens in enumerate(tokens.tolist()):
            # truncate output if EOS in prompt
            if self.tokenizer.eos_id() in prompt_tokens:
                eos_idx = prompt_tokens.index(self.tokenizer.eos_id())
                prompt_tokens = prompt_tokens[:eos_idx]
            out_tokens.append(prompt_tokens)
            out_texts.append(self.tokenizer.Decode(prompt_tokens))

        return (out_texts, out_tokens)        





if __name__ == '__main__':
    torch.manual_seed(0)
    use_cuda = False
    device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"

    prompts = ["Opeth's best album is", "Translate English to Swedish: Heart in Hand =>", "Tell me more about the Israel-Palestine conflict"]

    model = LLaMA.build(
        chkpt_dir='llama-2-7b/',
        tok_pth='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=3,
        device=device
    )

    print("Loaded model")
    out_texts, out_tokens = model.text_completion(prompts, max_gen_len=128)
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(out_texts[i])
        print('*' * 50)