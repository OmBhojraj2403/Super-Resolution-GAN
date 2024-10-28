## 1. Data
In this repo, the [DIV2K dataset](http://data.vision.ee.ethz.ch/cvl/) was used, which includes: 

- 1600 training images:
    - 800 high resolution (HR) images (2K)
    - 800 respective low resolution images (LR, 4x downscale)

- 200 test images:
    - 100 HR
    - 100 LR

## 2. Repo structure

```
.
├── CONFIG.py
├── DIV2K
│   ├── DIV2K_train_HR
│   ├── DIV2K_train_LR_bicubic
│   │   └── X4
│   ├── DIV2K_valid_HR
│   └── DIV2K_valid_LR_bicubic
│       └── X4
├── README.md
├── requirements.txt
├── model_summary.txt
├── psnr_ssim_valid.txt
├── models
├── config.py
├── dataloader.py
├── discriminator.py
├── evaluate.py
├── generate_super_res.py
├── generator.py
├── perceptual_loss.py
├── train.py
├── utils.py
└── valid_results
```

- `DIV2K`: consists of our data:
    - `DIV2K_train_HR`: 800 HR training images
    - `DIV2K_train_LR_bicubic/X4`: 800 LR training images
    - `DIV2K_valid_HR`: 100 HR test images
    - `DIV2K_valid_LR_bicubic/X4`: 100 LR test images
- `psnr_ssim_valid.txt`: PSNR and SSIM values for the validation data
- `config.py`: configurations for data, models and training.
- `dataloader.py`: custom torch dataset for DIV2K with functions for loading the data
- `discriminator.py`: contains the discriminator
- `evaluate.py`: evaluating script
- `generate_super_res.py`: script to generate and save super resolution images
- `generator.py`: contains the generator
- `perceptual_loss.py`: loss functions for generator
- `train.py`: training script
- `valid_results:` contains validation results of the trained model

## 3. Usage
```commandline
cd src/
```

- Training:
```commandline
python train.py
```

- Inference: 
```commandline
python generate_super_res.py
```

- Evaluate PSNR and SSIM (change the SR_VAL_DIR value in generate_super_res.py if needed):
```commandline
python evaluate.py
```

## Model sizes:

- `Discriminator`: 59.7 MB
- `Generator`: 18.2 MB

## 4. Results

| Original High Resolution images      | GAN-Generated Super Resolution images |
| --------------------------- | --------------------------------- |
| ![](readme_utils/0859.png)      | ![](valid_results/sr_X4_0859x4.png)       |
| ![](readme_utils/0855.png)      | ![](valid_results/sr_X4_0855x4.png)       |
| ![](readme_utils/0891.png)      | ![](valid_results/sr_X4_0891x4.png)       |
| ![](readme_utils/0863.png)      | ![](valid_results/sr_X4_0863x4.png)       |
| ![](readme_utils/0878.png)      | ![](valid_results/sr_X4_0878x4.png)       |


