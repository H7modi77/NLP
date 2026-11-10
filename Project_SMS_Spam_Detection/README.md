# SMS Spam Detection - TF-IDF vs Sentence-BERT

ARTI 404 (NLP) - Section 7MA2 - Group 9
Imam Abdulrahman Bin Faisal University, College of Computer Science and IT
Instructor: Dr. Mohammed Mosuily

| Name | ID |
|---|---|
| Mohammed Nasser Almarhoon (team leader) | 2240005893 |
| Ali Zuhair Alsaffar | 2240005706 |

## What it does
Input: one short English SMS. Output: **Spam** or **Not spam**.

We compare two methods on the UCI SMS Spam Collection:
- **Method 1:** TF-IDF (word unigrams + bigrams) + Logistic Regression
- **Method 2:** frozen `all-MiniLM-L6-v2` Sentence-BERT embeddings + Logistic Regression

Our question: *which one detects misspelled / obfuscated spam (fr33, w1n, cl4im, txt, msg) more reliably?*

## How to run
Open `SMS_Spam_Detection_Group9.ipynb` in Google Colab and run all cells (Runtime → Run all). It downloads the dataset and the model by itself, takes a few minutes on the free CPU runtime. The **Try it** section at the end lets you type any SMS and classifies it without retraining.

Locally: `pip install -r requirements.txt` then run the notebook in Jupyter.

## Setup
- Cleaning: removed empty rows and exact duplicates (5,574 → 5,160 messages, 4,518 ham / 642 spam)
- Split 70/15/15 stratified, seed 42 → train 3,612 / dev 774 / test 774
- `class_weight='balanced'`, hyperparameter `C` (and `min_df` for TF-IDF) picked on dev only, test used once

## Results (test set)
| | TF-IDF + LR | SBERT + LR |
|---|---|---|
| Accuracy (full test) | **0.9845** | 0.9780 |
| Macro F1 (full test) | **0.9640** | 0.9501 |
| Macro F1 - challenge subset (205 msgs) | 0.9538 | **0.9694** |
| Macro F1 - obfuscated only (184 msgs) | 0.9598 | **0.9829** |
| Spam recall when 50% of letters are leet-replaced | 0.823 | **0.969** |

TF-IDF is slightly better overall, but SBERT is more reliable on obfuscated messages. Full error analysis and discussion are in the notebook.

## Repo contents
- `SMS_Spam_Detection_Group9.ipynb` - the full notebook (with outputs)
- `requirements.txt`
- `README.md`
- `Proposal_Group9.pdf` - the approved project proposal

## References
[1] T. A. Almeida, J. M. G. Hidalgo, and A. Yamakami, "Contributions to the study of SMS spam filtering: new collection and results," Proc. DocEng, 2011, pp. 259-262.
[2] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," Proc. EMNLP-IJCNLP, 2019, pp. 3982-3992.
