# Benchmarking In-the-Wild Multimodal Plant Disease Recognition and A Versatile Baseline
Official implementation of ['Benchmarking In-the-Wild Multimodal Plant Disease Recognition and A Versatile Baseline'](https://arxiv.org/abs/2408.03120)

The paper has been accepted by **ACM Multimedia 2024**. 


## Introduction
We curate an in-the-wild multimodal plant disease recognition dataset PlantWild with the largest number of disease classes. We introduce descriptive prompts in our dataset to provide rich information in textual modality. In addition, we propose a strong yet versatile baseline, which models text descriptions and visual data through multiple prototypes and can achieve outstanding performance on in-the-wild plant disease images.

### Curation of our dataset
<div align="center">
  <img width=800 src="figures/cleaning.png"/>
</div>
<div align="center">
  <img width=800 src="figures/bar.png"/>
</div>

### Workflow of the baseline
<div align="center">
  <img width=500 src="figures/baseline.png"/>
</div>


## Preparation
### 1. Clone the repo
```bash
git clone https://github.com/tqwei05/MVPDR.git
cd MVPDR
```
### 2. Requirements
```bash
conda create -n mvpdr python=3.8
conda activate mvpdr
pip install -r requirements.txt
# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
```

### 3. Prepare the data
Our dataset is accessible through [PlantWild](https://drive.google.com/file/d/1s7FOoztTHvO03yVfw75pQY_kzZqvAckD/view?usp=drive_link).


## Running
We provide the code for training and evaluation in main.py.
```python
python main.py --config <CONFIG_DIR>
```

## Results
<div align="center">
  <img width=800 src="figures/results.png"/>
</div>


## Citation
If you find our work useful, please cite as follows.

```BibTeX
@inproceedings{MVPDR,
      title={Benchmarking In-the-Wild Multimodal Plant Disease Recognition and A Versatile Baseline},
      author={Wei, Tianqi and Chen, Zhi and Huang, Zi and Yu, Xin},
      booktitle={ACM Multimedia},
      year={2024}
}
```

## Acknowledgments

Our code benefits from [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter). We appreciate their wonderful works.



