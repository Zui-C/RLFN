# GPT-4v Evaluation on CROHME 2014 Dataset

We have conducted an evaluation of GPT-4v using the CROHME 2014 dataset. The results were meticulously recorded from October 8, 2023, to October 17, 2023.

## Data Files

- `GPT-4v.csv`: This file includes the ground truth, raw GPT-4v outputs, and processed results aligned with CROHME standards.
- `gpt4v.txt`: Contains solely the GPT-4v processed results.

## Experiment Details

Experiments were conducted manually by uploading images and collating results, as an API was not available. The prompt is 'generate LaTeX code and output without compile'. This prompt is designed to ensure the extraction of LaTeX code as opposed to plain text or rendered LaTeX.

The raw output from GPT-4v, presented in a free style, was transformed into the standardized CROHME style. For example:

- 'a^1' was converted to 'a^{1}'
- '\mathrm{E}' was converted to 'E'
- '\Pi' was converted to '\pi'
- '\to' was converted to '\rightarrow'
- '\left(' was converted to '('
- '\mid' was converted to '|'
- '\,' is deleted

All LaTeX command names appearing in the output were treated as single characters and were separated by spaces.

## Experiment Results

| **CROHME2014**      | **ExpRate** | **Error1** | **Error2** |
| ------------------- | ----------- | ---------- | ---------- |
| **DWAP** (baseline) | 51.72       | 69.47      | 77.99      |
| **RLFN** (ours)     | 57.00       | 72.01      | 80.73      |
| **GPT-4v**          | 31.85       | 49.09      | 60.45      |

## Discussions

- GPT-4v was not fine-tuned, leading to a noticeable out-of-vocabulary issue. 
- GPT-4v demonstrated enhanced capabilities, attributed to its pretraining on a vast dataset.

- GPT-4v exhibited a tendency for hallucination, a characteristic shared among large language models (LLMs). This observation underscores potential challenges associated with the broader application of pretrained LLMs.

- GPT-4v's performance consistency is anticipated to vary. 

## Acknowledgements

The majority of the experimental work and data processing was executed by Jiaqi Han.