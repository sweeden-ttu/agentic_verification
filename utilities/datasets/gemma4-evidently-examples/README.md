# gemma4-evidently-examples

The Kaggle dataset `scottweeden/gemma4-evidently-examples`, the evaluation
baseline that `modules/knowledge-improvement-loop` reads.

**The data is not committed yet.** The dataset is private, and the Kaggle API
refused anonymous access when this folder was created. To bring it in:

```bash
./fetch.sh          # needs ~/.kaggle/kaggle.json, or KAGGLE_USERNAME and KAGGLE_KEY
git add data && git commit -m "data: add gemma4-evidently-examples"
```

`fetch.sh` copies from `/kaggle/input/datasets/scottweeden/gemma4-evidently-examples`
when it runs inside Kaggle, and downloads through the Kaggle API elsewhere.

Before committing, check the file sizes. GitHub rejects files over 100 MB, so
larger files need Git LFS or DVC. Check the dataset's license allows
redistribution, too, if this repository is public.

Point the loop at it with:

```bash
export KAGGLE_BASELINE_DIR=$(pwd)/data
```
