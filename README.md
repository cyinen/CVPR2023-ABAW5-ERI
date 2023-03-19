# requirement

```
pip install -r requirements.txt

```



# train the model
```shell
python main.py --gpus 1 \
            --auto_lr_find \
            --d_model 256 \
            --devices 1\
            --accelerator gpu \
            --gradient_clip_val 0.1\
            --stage fit \
            --lr 1e-4 \
            --feature  DeepSpectrum PosterV2+Vit
```

