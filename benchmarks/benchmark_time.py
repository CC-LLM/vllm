import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Union

from vllm import LLM, SamplingParams, RequestOutput, EmbeddingRequestOutput


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B", help="model name or path")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--input-len", type=int, default=128, help="input seq len")
    parser.add_argument("--output-len", type=int, default=128, help="output seq len")
    parser.add_argument("--warmup-times", type=int, default=10, help="warmup times")
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="sampling top p")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    return args


class LLMProfiler(LLM):

    def init_data(self, input_len: int = 128, output_len: int = 128, warm_up_times: int = 10):
        self.start_time = None
        self.prefill_time = None
        self.end_time = None
        self.input_len = input_len
        self.output_len = output_len
        self._warm_up(warm_up_times)

    def _warm_up(self, warm_up_times: int = 10):
        self.is_warmed_up = False
        self.generate(['hi!'] * warm_up_times, SamplingParams(temperature=1.0, top_p=1.0))
        self.is_warmed_up = True
        self.prefill_time = None

    def _is_stop(self, output: RequestOutput):
        if self.is_warmed_up:
            return len(output.outputs[0].token_ids) == self.output_len
        else:
            return output.finished

    def _run_engine(
            self, use_tqdm: bool
    ) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=f"Generation Speed: {0:.2f} toks/s",
            )
        # Run the engine.
        outputs: List[Union[RequestOutput, EmbeddingRequestOutput]] = []
        total_toks = 0
        self.start_time = time.perf_counter()
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            if self.prefill_time is None:
                self.prefill_time = time.perf_counter()
            for output in step_outputs:
                if self._is_stop(output):
                    outputs.append(output)
                    if use_tqdm:
                        if isinstance(output, RequestOutput):
                            # Calculate tokens only for RequestOutput
                            total_toks += sum(
                                len(stp.token_ids) for stp in output.outputs)
                            spd = total_toks / pbar.format_dict["elapsed"]
                            pbar.postfix = f"Generation Speed: {spd:.2f} toks/s"
                        pbar.update(1)
        self.end_time = time.perf_counter()
        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs


def main(args):
    llm = LLMProfiler(model=args.model)
    prompts = None
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p)
    vocab_size = len(llm.llm_engine.tokenizer.tokenizer.vocab)
    prompt_token_ids = np.random.randint(0, vocab_size, size=(args.batch_size, args.input_len)).tolist()
    llm.init_data(args.input_len, args.output_len, args.warmup_times)
    llm.generate(prompts, sampling_params, prompt_token_ids)

    print(f'prefill cost : {llm.prefill_time - llm.start_time:.4f}s')
    print(f'generate time: {llm.end_time - llm.prefill_time:.4f}s')
    print(f'total cost   : {llm.end_time - llm.start_time:.4f}s')


if __name__ == '__main__':
    main(parse_args())
