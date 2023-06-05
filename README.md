# realtime-chatbot
**A Full-Duplex Open-Domain Dialogue Agent with Continuous Turn-Taking Behavior**

To reproduce results from the paper, see [Reproduce Paper Results](#reproduce-paper-results).

Inspired by [Google Duplex](https://ai.googleblog.com/2018/05/duplex-ai-system-for-natural-conversation.html), this bot aims to provide an experience as close as possible to a live phone call or face-to-face conversation. Unlike Google Duplex which was designed
for specific tasks, this is a completely open-domain system intended to converse about anything. Importantly, there are no pre-defined
turn-taking rules - the agent is free to speak whenever it chooses and learns coordination behavior directly from the training data.

- [Whisper](https://github.com/openai/whisper) is used for Automatic Speech Recognition (ASR).
- [OPT 2.7b](https://huggingface.co/facebook/opt-2.7b) fine-tuned on transcribed spoken dialogue from [TalkBank](https://ca.talkbank.org/access/) is used for the dialogue agent. See the [model card](https://huggingface.co/AbrahamSanders/opt-2.7b-realtime-chat-v2) for more details.
- [FastSpeech2](https://huggingface.co/facebook/fastspeech2-en-200_speaker-cv4) (trained on [Common Voice v4](https://commonvoice.mozilla.org/en/datasets)) or [Bark](https://github.com/suno-ai/bark) (trained on _yet to be published_) is used for Text to Speech (TTS).

![System architecture](images/system_architecture.png)

## Installation
### Dependencies
Python 3.8 or greater is required. If PyTorch is not already installed in your environment, please install 
the appropriate configuration of PyTorch for your environment (OS, CUDA version) before proceeding - 
see [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

If you wish to use Bark for TTS, the nightly PyTorch build (2.1.x) offers additional performance improvements. Otherwise, the latest stable release is recommended.
See the [Bark Readme](https://github.com/suno-ai/bark#%EF%B8%8F-hardware-and-inference-speed) for more details.

To clone the repo and install dependencies, run:
```bash
git clone https://github.com/AbrahamSanders/realtime-chatbot.git
cd realtime-chatbot
pip install -r requirements.txt
```

## Run Chat Interfaces
### Gradio Web Interface (audio + text)
To launch the Gradio web interface, run the following. When prompted, navigate to [http://127.0.0.1:7860](http://127.0.0.1:7860):
```bash
python run_gradio.py
```
By default, FastSpeech2 is used for TTS. To use Bark instead, run:
```bash
python run_gradio.py --tts-engine=bark
```

Running this interface will use between 12GB and 24GB of GPU RAM, depending on the selected Whisper model size.
Under default settings, it should run smoothly on a machine with a single 16GB GPU, with either FastSpeech2 or Bark, 
however you may experience larger floor transfer offsets (response latencies) on this minimal hardware configuration.

If you have multiple GPUs, the system will attempt to distribute the models across devices for added performance:
- If two GPUs are available, one will run the agent (OPT 2.7b) and the other will run Whisper and FastSpeech2 / Bark.
- On a machine with three or more GPUs, OPT, Whisper, and FastSpeech2 / Bark will each run on their own dedicated GPU to maximize performance.

Audio input and output devices (microphone + speakers) are required. There is currently no built-in echo cancellation functionality,
so for the best experience it is recommended to use:
- A high-quality headset.
- Alternatively, headphones and an external microphone.

#### Interface Usage:

After the interface loads:
1. Click Record to allow Gradio to begin recording audio from your microphone.
2. [Optional] Use the `Dialogue Summary Prompt` textbox to provide a short script to help guide the topic and structure of the conversation.
    - e.g., `"S1 and S2 are talking about what's new in their lives. S2 got a new dog."`
    - If set to a blank string, the conversation will be completely open-ended.
3. [Optional] Use the `Agent Starts` checkbox to determine whether the agent will start the conversation or wait for the user to speak first.
    - If `Agent Starts` is checked, use the `Opening Utterance` textbox to provide the agent's initial utterance. If set to a blank string, the agent will be
free to start the conversation however it chooses.
4. [Optional] Use the `Agent Voice` dropdown (scroll to bottom of page) to select the voice used by the agent.
    - Other options exist nearby to customize the agent's persona, such as `Agent Name`, `Agent Age`, and `Agent Gender`.
5. Uncheck `Reset` to begin the conversation.
6. To reset the conversation at any time, check and then uncheck `Reset`.

![Gradio web interface](images/gradio_interface.png)

### Terminal Interface (text only)
To launch the terminal interface, run:
```bash
python run_chat.py
```

The purpose of the terminal interface is to provide a simple way to test the agent model in a text-only environment without the added complexity of ASR and TTS.

Keyboard input into the terminal input is processed in real-time to emulate continuous speech input.
While you type, words are submitted to the agent after `space` or `enter` are pressed.

- Type `--reset` to clear the dialogue history and start over.
- Type `--exit` to quit.

![Terminal interface](images/terminal_interface.png)

## Reproduce Paper Results
To reproduce the results in tables 4 & 5 in the paper:
1. Ensure `data/dataset_test.txt` exists (details on obtaining this TBD due to TalkBank corpora licenses)
2. Run the evaluation script:
```bash
python run_evals.py > eval_results_all.txt
```
This will run evaluation on all available GPUs using multiprocessing. On 4 GPUs with 48GB of memory each, this should take about ~13 hours.
On smaller GPUs, lower the `--batch-size` and `--contrastive-batch-size` as needed.
The results from table 4 will be saved to `evals_output_ppl_all.csv` and the results from table 5 will be saved to `evals_output_pred_all_all.csv`.

## Training
To train an agent model, first prepare the dataset and then run the HuggingFace trainer. Scripts are provided for both.

### Prepare the dataset
This script downloads, pre-processes and formats talkbank conversational corpora into text files for training, also handling separation into train, dev, and test splits. Simply run:
```bash
python prep_datast.py --standardize-pauses
```
The dataset files will be placed into the `data` folder.

It is also possible to specify individual talkbank corpora or change the default train/dev/test split. To do this, check the 
command line options:
```bash
python prep_datast.py --help
```

### Train an agent model
The [train.py](train.py) script is a modified copy of [HuggingFace's run_clm.py script](https://github.com/huggingface/transformers/blob/v4.24.0/examples/pytorch/language-modeling/run_clm.py), adapted to use with line-by-line text file datasets that require 
padding each example instead of chunking them into fixed size blocks.

The provided shell script [train_large.sh](train_large.sh) is pre-configured to fine-tune `facebook/opt-2.7b` using `train.py`. 
To fine-tune a different model, simply modify this script. For example to train `facebook/opt-350m` instead, modify it as such:

```bash
python train.py \
    --model_name_or_path=facebook/opt-350m \
    ...
```