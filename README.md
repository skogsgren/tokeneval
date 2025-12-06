# Setup

Clone this repo and cd in:

```
git clone https://github.com/skogsgren/tokeneval
cd ./tokeneval
```

Clone the necessary repos:

```
git clone https://github.com/LeonidasY/fast-vocabulary-transfer/tree/main/fvt
git clone https://github.com/arcee-ai/mergekit ./mergekit-git
```

Install the necessary dependencies:

```
pip install -U pip &&
    pip install --no-cache-dir -r ./requirements.txt &&
    pip install -e mergekit-git/
```

You also need a valid installation of torch, but seeing as it's torch and
there's always something about the installation of torch I'll leave that to
your discretion.

# Running

The main pipeline is kind of hardcoded in that it expects a data-structure like
the one used in Goldfish, like so:

```
salad/data/
├── train
│   ├── arb_arab.txt
│   ├── dan_latn.txt
│   ├── ell_grek.txt
│   ├── eng_latn.txt
│   ├── est_latn.txt
│   ├── isl_latn.txt
│   ├── nob_latn.txt
│   ├── swe_latn.txt
│   └── tur_latn.txt
└── valid
    ├── arb_arab.txt
    ├── dan_latn.txt
    ├── ell_grek.txt
    ├── eng_latn.txt
    ├── est_latn.txt
    ├── isl_latn.txt
    ├── nob_latn.txt
    ├── swe_latn.txt
    └── tur_latn.txt

3 directories, 18 files
```

So if you want to run the experiments, download the OSCAR data, and sample
using the `./sample_dataset.py` script, e.g:

```
./sample_dataset.py <input_file> <byte_premium_file> <output_file>
```

For the byte premium file, see `./byte_premium_goldfish.yaml`

---

The configuration file for the pipeline is in YAML, like so:

```{yaml}
paths:
  tokenization_transfer_out: ./salad/transfer
  mono_finetuned_out: ./salad/finetuned_mono
  mixed_finetuned_out: ./salad/finetuned_mixed

  # directory to files created using the sample_dataset.py file
  goldfish_datadir: ./salad/data
  flores_dir: ./data/flores/

languages:
  - eng_Latn
  - est_Latn
  - tur_Latn
  - swe_Latn
  - nob_Latn
  - dan_Latn

# which model sizes to use for the source/target models
models:
  source_size: 100MB
  target_size: 10MB

trainargs:
  device: cuda
  steps: 750
  batch_size: 8
  learning_rate: 0.0001
  weight_decay: 0.0001
```

After that you can run the pipeline. Because CUDA is CUDA and dependencies are
hell, I opted for Docker. See the scripts under `./docker` if you're the same.
They're basically just wrappers around this command:

```
python3 pipeline.py /path/to/cfg.yaml
```

**NOTE**: there's still some hardcoded elements to CUDA, so if you're running
this on CPU (may God have mercy on your soul), you'll probably have to go in to
the source code sometimes and change it from cuda to cpu.
