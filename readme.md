# COMP0197 Group Coursework Code

This repository contains code for finetuning and pretraining models.

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

## Pretraining

To pretrain the model, follow these steps:

1. Prepare the coco dataset and save it as `./coco_dataset/raw` in the main directory

2. Run the pretraining script:

    ```shell
    python pretrain.py --mask grid --masking-ratio 0.2 --directory `saved_models/pretrain/`
    ```

    flags:
    - --mask : the masking method for the pretrained model (only works if pretrain is true)
    - --masking-ratio : the ratio of the pretrainined model ((only works if pretrain is true))
    - --directory : The location of the saved directory
    - --load-checkpoint : Load a saved pretrained model for further training 
        - (looks in --directory for the folder and takes the last epoch)

## Finetuning

To finetune the model, follow these steps:

1. Download the pre-trained model weights.

2. Prepare your dataset in the required format.

3. Run the finetuning script:

    ```shell
    python finetune.py < /path/to/output_logs.txt
    ```
    flags:
    - --pretrain : to set pretraining to true
    - --mask : the masking method for the pretrained model (only works if pretrain is true)
    - --masking-ratio : the ratio of the pretrainined model ((only works if pretrain is true))
    - --patience : The amount of none improvements epochs before the training is stopped 

    Replace `/path/to/output_logs.txt` with the path to where the training logs should be outputted

## Contributing

Contributions are welcome! Please follow the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE).