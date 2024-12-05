## This Repo is for CSE 256 LIGN 256 - Statistical Natural Lang Proc - Nakashole [FA24] PA2
### Author: [Zhecheng Li](https://github.com/Lizhecheng02) && Professor: [Ndapa Nakashole](https://ndapa.us/)

### Python Environment

#### 1. Install Packages

```b
pip install -r requirements.txt
```

### Prepare Data
All datasets are already in the GitHub repo.

### Run Codes
##### 1. Encoder
- If you want to train with ``traditional attention`` and ``mean embedding output``, use:

  ```bas
  python main.py --run "encoder_classic_mean"
  ```

- If you want to train with ``slide window attention`` and ``mean embedding output``, use:

  ```bas
  python main.py --run "encoder_window_attention"
  ```
  
- If you want to train with ``alibi relative positional embedding`` and ``mean embedding output``, use:

  ```ba
  python main.py --run "encoder_alibi"
  ```
  
- If you want to train with ``disentangled attention patterns`` and ``mean embedding output``, use:

  ```bas
  python main.py --run "encoder_deberta"
  ```
  
- If you want to train with extra **[cls]** token to represent the final embedding output, use:

  ```bas
  python main.py --run "encoder_cls_token"
  ```


You can change the parameters in ``main.py``, but you should be able to get around 86-87% accuracy using default values.

<div style="text-align: center;">
    <img src="./acc_plots/encoder_classic_mean_acc.png" width="75%" />
</div>

##### 2. Decoder

- If you want to train the traditional decoder-only model for text generation, use:

  ```bas
  python main.py --run "decoder"
  ```

You can also change the parameters in ``main.py``, but you should be able to get around 4.8 loss using default values.

<div style="text-align: center;">
    <img src="./acc_plots/decoder_loss.png" width="75%" />
</div>

### Questions

You are welcome to discuss any issues you encounter while running this GitHub repository. Feel free to either open an issue or contact me directly at **zhl186@ucsd.edu**.