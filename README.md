# Object-aware Adaptive-Positivity Learning for Audio-Visual Question Answering(AAAI'2024) [[arXiv](https://arxiv.org/abs/2312.12816)]

Authors: [Zhangbin Li](https://github.com/zhangbin-ai), Dan Guo, [Jinxing Zhou](https://github.com/jasongief), Jing Zhang, and Meng Wang
---
## Requirements

```python
python3.7 +
pytorch1.7.1
numpy
ast
```


## Usage

1. **Cloning this repo**

   ```python
   git clone https://github.com/zhangbin-ai/APL.git
   ```


2. **Getting Started**

    + Training
        ```python
        python train.py \
        --batch-size 64 \
        --epochs 30 \
        --lr 1e-4 \
        --gpu 0 \
        --checkpoint APL_dir \
        --checkpoint_file checkpoint01 \
        --save_model_flag True \
        ```

    + Testing
        ```python
        python test.py
        ```

3. **Citing Us**

    If you find this work useful, please consider citing it.
    ```
    Coming soon.
    ```

4. **Acknowledgement**
    The computation is supported by the HPC Platform of Hefei University of Technology.
