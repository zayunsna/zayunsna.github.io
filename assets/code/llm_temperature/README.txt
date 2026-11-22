Temperature-zero and repeated-sampling measurements, 2026-11-23.

Environment: Python 3.12.13, torch 2.14.0, transformers 5.17.0, CPU bfloat16,
eager attention, four CPU threads, macOS arm64.
Model: Qwen/Qwen2.5-0.5B-Instruct
Revision: 7ae557604adf67be50417f59c2c2f167def9a775
The original article did not pin a revision; this is the existing cached snapshot,
also recorded in the project's experiment notes, now pinned for reproducibility.

Install requirements in a fresh environment:
  python -m pip install -r requirements.txt
Run from the directory containing sampling_measurements.py:
  python sampling_measurements.py --output-dir results > stdout.txt 2> stderr.txt
To run the article's added blocks, keep this script beside the code being run.

The model and tokenizer must already be in the Hugging Face cache. Loading uses
local_files_only=True and offline flags; missing files cause an error.
No hosted inference, API key or model download is used by this script.

reference_output.txt is the complete stdout of the published script.
reference_stderr.txt is the complete stderr, including loading progress and
the warnings about sampling-only flags in greedy generation.
sampling_results.json records the environment, cached model generation defaults,
explicit experiment settings, summaries and every sampled token ID/text.
Loading progress rates in stderr are incidental and vary by run.

Protocol: raw prompt "My favorite programming language is", no chat template,
temperatures 0.3/1.0/2.0, 20 separate calls each, seeds 0 through 19 reset per
temperature, max_new_tokens=12, repetition_penalty=1.0, top_k=0, top_p=1.0,
num_beams=1, use_cache=True, default model EOS IDs, tokenizer EOS as pad token.
Each call stops at an EOS token or at the new-token limit.
Unique continuations compare complete generated token-ID tuples after removing
the prompt; a generated EOS is retained. Decoded text skips special tokens.
First-token counts use the first generated token ID, including the leading space
in the decoded token " Python". No answer quality or correctness metric is used.

The zero-temperature check expects a ValueError with sampling enabled and compares
greedy temperature=0 against greedy temperature=1 on the same prompt/model.
The generate helper copies GenerationConfig before each call so that only
do_sample/temperature overrides differ and other model defaults remain controlled.

These counts cover one prompt, one small model and 20 seeded samples per temperature.
The sample-size ceiling limits interpretation when all 20 outputs are different.
Other models, other hardware backends, chat-template prompts and filters were not
measured for this addition. Existing article experiments remain unchanged.
