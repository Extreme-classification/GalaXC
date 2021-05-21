# GalaXC
## [GalaXC: Graph Neural Networks with Labelwise Attention for Extreme Classification](http://manikvarma.org/pubs/saini21.pdf)
```bib
@InProceedings{Saini21,
	author       = {Saini, D. and Jain, A.K. and Dave, K. and Jiao, J. and Singh, A. and Zhang, R. and Varma, M.},
	title        = {GalaXC: Graph Neural Networks with Labelwise Attention for Extreme Classification},
	booktitle    = {Proceedings of The Web Conference},
	month = "April",
	year = "2021",
	}
```

#### Setup GalaXC
```bash
git clone https://github.com/Extreme-classification/GalaXC.git
conda env create -f GalaXC/environment.yml
conda activate galaxc
pip install hnswlib
git clone https://github.com/kunaldahiya/pyxclib.git
cd pyxclib
python setup.py install
cd ../GalaXC
```

#### Dataset Structure
Your dataset should have the following structure:
```
DatasetName (e.g. LF-AmazonTitles-131K)
│   trn_X.txt   (text for trn documents, one text in each line)
|   tst_X.tst   (text for tst documents, one text in each line)
|   Y.txt       (text for labels, one text in each line)
│   trn_X_Y.txt (trn labels in spmat format)
|   tst_X_Y.txt (tst labels in spmat format)
|   filter_labels_test.txt (filter labels where label and test documents are same)
│
└───XXCondensedData (embeddings for tst, trn documents and labels, for benchmark datasets, XX=DX[Astec])
    │   trn_point_embs.npy (2D numpy matrix for trn document embeddings)
    │   tst_point_embs.npy (2D numpy matrix for tst document embeddings)
    |   label_embs.npy     (2D numpy matrix for label embeddings)

```

We have provided the DX(embeddings from Module 1 of Astec) embeddings for public benchmark datasets for ease of use. Got better(higher recall) embeddings from somewhere? Just plug the new ones and GalaXC will have better preformance, no need to make any code change! These files for LF-AmazonTitles-131K, LF-WikiSeeAlsoTitles-320K and LF-AmazonTitles-1.3M can be found [here](https://drive.google.com/drive/folders/1PamOpzMV6NlgvBEOwpxPZ4dahnun-dtN?usp=sharing). Except the files in DXCondensedData, all other files are copy of the datasets from [The Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html).


#### Sample Runs
To reproduce the numbers on public benchmark datasets reported in the paper, the sample runs are 

**LF-AmazonTitles-131K** 
```bash
python -u -W ignore train_main.py --dataset /your/path/to/data/LF-AmazonTitles-131K --save-model 0  --devices cuda:0  --num-epochs 30  --num-HN-epochs 0  --batch-size 256  --lr 0.001  --attention-lr 0.001 --adjust-lr 5,10,15,20,25,28  --dlr-factor 0.5  --mpt 0  --restrict-edges-num -1  --restrict-edges-head-threshold 20  --num-random-samples 30000  --random-shuffle-nbrs 0  --fanouts 4,3,2  --num-HN-shortlist 500   --embedding-type DX  --run-type NR  --num-validation 25000  --validation-freq -1  --num-shortlist 500 --predict-ova 0  --A 0.6  --B 2.6
```

**LF-WikiSeeAlsoTitles-320K** 
```bash
python -u -W ignore train_main.py --dataset /your/path/to/data/LF-WikiSeeAlsoTitles-320K --save-model 0  --devices cuda:0  --num-epochs 30  --num-HN-epochs 0  --batch-size 256  --lr 0.001  --attention-lr 0.05 --adjust-lr 5,10,15,20,25,28  --dlr-factor 0.5  --mpt 0  --restrict-edges-num -1  --restrict-edges-head-threshold 20  --num-random-samples 32000  --random-shuffle-nbrs 0  --fanouts 4,3,2  --num-HN-shortlist 500  --repo 1  --embedding-type DX --run-type NR  --num-validation 25000  --validation-freq -1  --num-shortlist 500  --predict-ova 0  --A 0.55  --B 1.5
```

**LF-AmazonTitles-1.3M** 
```bash
python -u -W ignore train_main.py --dataset /your/path/to/data/LF-AmazonTitles-1.3M --save-model 0  --devices cuda:0  --num-epochs 24  --num-HN-epochs 15  --batch-size 512  --lr 0.001  --attention-lr 0.05 --adjust-lr 4,8,12,16,18,20,22  --dlr-factor 0.5  --mpt 0  --restrict-edges-num 5  --restrict-edges-head-threshold 20  --num-random-samples 100000  --random-shuffle-nbrs 1  --fanouts 3,3,3  --num-HN-shortlist 500   --embedding-type DX  --run-type NR  --num-validation 25000  --validation-freq -1  --num-shortlist 500 --predict-ova 0  --A 0.6  --B 2.6
```
