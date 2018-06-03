
# Contextualized Word Representations for Reading Comprehension
### Shimi Salant and Jonathan Berant
[https://arxiv.org/abs/1712.03609](https://arxiv.org/abs/1712.03609)

#### Requirements

[Theano](http://deeplearning.net/software/theano/install.html), [Matplotlib](http://matplotlib.org/), [Java](https://www.oracle.com/java/index.html)

#### Setup (1): Preparing SQuAD

```bash
$ python setup.py prepare-squad
```
Downloads GloVe word embeddings and [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)
(download will be skipped if [zipped GloVe file](http://nlp.stanford.edu/data/glove.840B.300d.zip) is manually placed in `data` directory).
Once downloaded, SQuAD's training and development sets will be pre-processed and tokenized.<br />


#### Setup (2): Preparing pre-trained LM

```bash
$ python setup.py prepare-lm
```
Downloads the [pre-trained language model](https://github.com/tensorflow/models/tree/master/research/lm_1b) released along [1].


#### Setup (3): Encoding SQuAD via the LM

Internal representations of the LM (when operated over SQuAD's questions and paragraphs) are calculated offline and saved to disk in shards.
In order to manufacture and persist a shard, execute:

```bash
$ python setup.py lm-encode --dataset DATASET --sequences SEQUENCES --layer LAYER --num_shards NUM_SHARDS --shard SHARD --device DEVICE 
```
Where `DATASET` is either `train` or `dev`; `SEQUENCES` is either `contexts` or `questions`; and `LAYER` is `L1`, `L2` or `EMB` corresponding to _LM(L1)_, _LM(L2)_ and _LM(emb)_ in the paper, respectively.

Since this is a lengthy process, it can be carried out in parallel if multiple GPUs are available: specify the number of shards to produce via `NUM_SHARDS`, the current shard to work on via `SHARD`, and the device to use via `DEVICE` (`cpu` or an indexed GPU specifications e.g. `gpu0`).

For example, in order to manufacture the first out of 4 shards via the first GPU when producing  _LM(L1)_ encodings for the training dataset's paragraphs, execute:

```bash
$ python setup.py lm-encode --dataset train --sequences contexts --layer L1 --num_shards 4 --shard 1 --device gpu0 
```


#### Training and Validation

```bash
$ python main.py --name NAME --mode MODE --lm_layer LM_LAYER --device DEVICE
```
Supply an arbitrary name as `NAME` (log file will be named as such), and set `MODE` to one of: `TR`, `TR_MLP` or `LM` which respectively correspond to _TR_, _TR(MLP)_ and to the LM-based variants from the paper.

If `LM` is chosen, specify the internal LM representation to utilize by setting `LM_LAYER` to one of: `L1`, `L2`, or `EMB`.


#### Results

Validation set:

| Model                      | EM   | F1   |
| -------------------------- |:----:| ----:|
| RaSoR (base model [2])     | 70.6 | 78.7 |
| RaSoR + TR(MLP)            | 72.5 | 79.9 |
| RaSoR + TR                 | 75.0 | 82.5 |
| RaSoR + TR + LM(emb)       | 75.8 | 83.0 |
| RaSoR + TR + LM(L1)        | 77.0 | 84.0 |
| RaSoR + TR + LM(L2)        | 76.1 | 83.3 |

Test set results available on [SQuAD's leaderboard](https://rajpurkar.github.io/SQuAD-explorer/).

---

Tested in the following environment:

* Ubuntu 14.04
* Python 2.7.6
* NVIDIA CUDA 8.0.44 and cuDNN 5.1.5
* Theano 0.8.2
* Matplotlib 1.3.1
* Oracle JDK 8

---

[1] Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. 2016. Exploring the limits of language modeling. CoRR abs/1602.02410

[2] Kenton Lee, Shimi Salant, Tom Kwiatkowski, Ankur P. Parikh, Dipanjan Das, and Jonathan Berant. 2016. Learning recurrent span representations for extractive question answering. CoRR abs/1611.01436.

