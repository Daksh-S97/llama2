from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer

class LLaMA:

    def __init__(self, model:Transformer, args:ModelArgs, tokenizer:SentencePieceProcessor) -> None:
        
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


if __name__ == '__main__':
    torch.manual_seed(0)
    use_cuda = False
    device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"

    model = LLaMA.build(
        chkpt_dir='llama-2-7b/',
        tok_pth='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=2,
        device=device
    )

    print("Loaded model")