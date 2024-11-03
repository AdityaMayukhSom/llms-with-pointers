### LLMs With Pointer

The following code was run on a Windows 11 desktop with NVIDIA RTX 4060 Ti with 8GB of VRAM and 32 GB of memory on an intel i7 13th generation processor. Additionally, this code can also be run on Google Colab with the provided `notebook.ipynb` file. 

#### 1. Install Scoop (Windows Package Manager)

Open PowerShell as Administrator and run:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex
```

#### 2. Install Pipx and Poetry

Install Pipx with Scoop:

```powershell
scoop install pipx
pipx ensurepath
```

Then install Poetry using Pipx:

```powershell
pipx install poetry
poetry config virtualenvs.in-project true
```

#### 3. Download the Dataset

1. Download the dataset from the [Google Drive link](https://drive.google.com/drive/folders/1djraMWEhzW7FOwfBG7IrnVsGkgZwn_fm?usp=sharing).
2. Create a `data/` folder in the project root and place the dataset files there.

#### 4. Get Hugging Face Access and API Key

1. Request access to the **LLaMA 3** models on Hugging Face if needed.
2. Generate an API key in Hugging Face (under **Settings** > **Access Tokens**).

#### 5. Set Up the .env File

1. Copy `.env.example` to `.env` in the project root.
2. Add your Hugging Face API key in `.env`:

#### 6. Run the Scripts

First, install the required packaged, then open the Poetry shell:

```powershell
poetry install
poetry shell
```

Then use the following commands:


| Task         | Command               |
| ------------ | --------------------- |
| **Train**    | `./scripts/train.ps1` |
| **Test**     | `./scripts/test.ps1`  |
| **Evaluate** | `./scripts/eval.ps1`  |


For evaluation with file input, follow instructions in `eval.ps1` to create an input file, add your data, then run the script.

### References

1. [DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models](https://arxiv.org/abs/2309.03883)
2. [Transformers and Pointer-Generator Networks for Abstractive Summarization](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/custom/15784595.pdf)
3. [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)