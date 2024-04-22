# COMP0197 Group Coursework Code

This repository contains code for finetuning and pretraining models.

## Environment

This code uses the default environment provided by `conda create -n comp0197-cw1-pt -c pytorch python=3.12 pytorch=2.2 torchvision=0.17` with no additional packages.

<!-- ## Prerequisites

- Python 3.x
- [PyTorch](https://pytorch.org/) (version X.X.X)
- [Transformers](https://huggingface.co/transformers/) (version X.X.X) -->

<!-- ## Installation

1. Clone the repository:

    ```shell
    git clone https://github.com/your-username/your-repo.git
    ```

2. Install the required dependencies:

    ```shell
    pip install -r requirements.txt
    ``` -->
# Instructions
## Pretraining

To pretrain the U-Net model you can run `python pretrain.py`. The Coco dataset must be downloaded manually with the raw image files in a directory called `./coco_dataset/raw`. The following parameters are also available:

```shell
usage: pretrain.py [-h] [-lc LOAD_CHECKPOINT] [-m {grid,pixel,random_erasing}] [-mr {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}] [-p PATIENCE] [-b BATCH_SIZE] [-dir SAVE_DIRECTORY]

options:
  -h, --help            show this help message and exit
  -lc LOAD_CHECKPOINT, --load-checkpoint LOAD_CHECKPOINT
                        Load checkpoint
  -m {grid,pixel,random_erasing}, --mask-method {grid,pixel,random_erasing}
                        Masking method to use
  -mr {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}, --mask-ratio {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
                        Masking ratio
  -p PATIENCE, --patience PATIENCE
                        Patience for early stopping
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size for training
  -dir SAVE_DIRECTORY, --save_directory SAVE_DIRECTORY
```

To pretrain the contrastive learning model you can run `python contrastive_learn_pretrain.py`. The Coco dataset must be downloaded manually with the raw image files in a directory called `./coco_dataset/raw`. The following parameters are also available:
```shell
usage: contrastive_learn_pretrain.py [-h] [-b BATCH_SIZE] [-lc LOAD_CHECKPOINT]

options:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size for training
  -lc LOAD_CHECKPOINT, --load-checkpoint LOAD_CHECKPOINT
                        Load checkpoint
```
## Finetuning

To finetune the model U-Net model from scratch you can run `python finetune.py`. The Oxford-IIIT Pets Dataset wil be automatically downloaded with this command. The following parameters are also available if you wish to finetune a pretrained model:

```shell
usage: finetune.py [-h] [-m {grid,pixel,random_erasing}] [-mr {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}] [-pt | --pretrain | --no-pretrain] [-p PATIENCE]
                   [-s {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}] [-b BATCH_SIZE] [-lc LOAD_CHECKPOINT]

options:
  -h, --help            show this help message and exit
  -m {grid,pixel,random_erasing}, --mask-method {grid,pixel,random_erasing}
                        Masking method to use
  -mr {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}, --mask-ratio {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
                        Masking ratio
  -pt, --pretrain, --no-pretrain
                        Use pretrained model
  -p PATIENCE, --patience PATIENCE
                        Patience for early stopping
  -s {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}, --dataset-size {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}
                        Fine-tuning dataset size ratio
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size for training
  -lc LOAD_CHECKPOINT, --load-checkpoint LOAD_CHECKPOINT
                        Load checkpoint
```

To finetune the model contrastive learning model you can run `python contrastive_learn_finetune.py` with the following parameters:
```shell
usage: contrastive_learn_finetune.py [-h] [-pt | --pretrain | --no-pretrain] [-p PATIENCE] [-s {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}] [-b BATCH_SIZE] [-lc LOAD_CHECKPOINT]

options:
  -h, --help            show this help message and exit
  -pt, --pretrain, --no-pretrain
                        Use pretrained model
  -p PATIENCE, --patience PATIENCE
                        Patience for early stopping
  -s {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}, --dataset-size {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}
                        Fine-tuning dataset size ratio
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size for training
  -lc LOAD_CHECKPOINT, --load-checkpoint LOAD_CHECKPOINT
                        Load checkpoint
```



<!-- ## Contributing

Contributions are welcome! Please follow the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md). -->
<!-- 
## License

This project is licensed under the [MIT License](LICENSE). -->