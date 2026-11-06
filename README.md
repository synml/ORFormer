# ORFormer (WACV 2025 Oral)

This is the offcial repository of WACV 2025 Oral paper **_ORFormer: Occlusion-Robust Transformer for Accurate Facial Landmark Detection_**.

[Project Website](https://ben0919.github.io/ORFormer/)

[arxiv Paper](https://arxiv.org/abs/2412.13174)

[WACV Paper](https://openaccess.thecvf.com/content/WACV2025/papers/Chiang_ORFormer_Occlusion-Robust_Transformer_for_Accurate_Facial_Landmark_Detection_WACV_2025_paper.pdf)

## Architecture

![Overview](img/ORFormer.png)

## Quantitative Results

![Results](img/quantitative.png)

## Training Commands

All experiments can be run on a NVIDIA GTX1080Ti(11Gb RAM).

### Quantized Heatmap Generator

```bash
#WFLW
CUDA_VISIBLE_DEVICES=x python train_heatmap_generator.py --dataset WFLW --name [run_name] --resultDir [result_dir] --lr 0.0007 --batch_size 128 --alpha 100
#300W
CUDA_VISIBLE_DEVICES=x python train_heatmap_generator.py --dataset 300W --name [run_name] --resultDir [result_dir] --lr 0.0008 --batch_size 128 --alpha 1000
#COFW
CUDA_VISIBLE_DEVICES=x python train_heatmap_generator.py --dataset COFW --name [run_name] --resultDir [result_dir] --lr 0.0007 --batch_size 128 --alpha 100
```

### ORFormer

```bash
#WFLW
CUDA_VISIBLE_DEVICES=x python train_ORFormer.py --dataset WFLW --name [run_name] --resultDir [result_dir] --lr 0.0001 --batch_size 64 --alpha 50 --vit ORFormer
#300W
CUDA_VISIBLE_DEVICES=x python train_ORFormer.py --dataset 300W --name [run_name] --resultDir [result_dir] --lr 0.0001 --batch_size 64 --alpha 100 --vit ORFormer
#COFW
CUDA_VISIBLE_DEVICES=x python train_ORFormer.py --dataset COFW --name [run_name] --resultDir [result_dir] --lr 0.0001 --batch_size 64 --alpha 50 --vit ORFormer
```

### Integration with FLD Methods

```bash
#WFLW
CUDA_VISIBLE_DEVICES=x python train_HGNet_with_ORFormer.py --dataset WFLW --name [run_name] --resultDir [result_dir] --lr 0.001 --batch_size 16 --alpha 0.05 --heatmap ORFormer
#300W
CUDA_VISIBLE_DEVICES=x python train_HGNet_with_ORFormer.py --dataset 300W --name [run_name] --resultDir [result_dir] --lr 0.001 --batch_size 16 --alpha 0.05 --heatmap ORFormer
#COFW
CUDA_VISIBLE_DEVICES=x python train_HGNet_with_ORFormer.py --dataset COFW --name [run_name] --resultDir [result_dir] --lr 0.001 --batch_size 16 --alpha 0.05 --heatmap ORFormer
```

### Determinism

All train and test scripts seed Python, NumPy and PyTorch with `SEED` in `Config/default.py` (default 0) and run PyTorch in strict deterministic mode, so the same seed gives bit-identical checkpoints on the same hardware and package versions. An op without a deterministic CUDA kernel raises an error instead of silently varying. To get a different run, change `SEED`.

## Evaluation

The test scripts evaluate the released weights in `weights/` by default. Pass `--weights` to evaluate your own checkpoint (ORFormer and integration only):

```bash
python test_ORFormer.py --dataset WFLW --weights [result_dir]/[run_dir]/best_model.pt
python test_HGNet_with_ORFormer.py --dataset WFLW --weights [result_dir]/[run_dir]/best_model.pt
```

Scripts change into the repository directory on start, so relative paths are resolved from there.

## ONNX Export

Exports HGNet + ORFormer as a single model. Inputs: `img256` (1, 3, 256, 256) and `img64` (1, 3, 64, 64), RGB face crops with ImageNet normalization. Output: `landmarks` (1, N, 2) in 256x256 crop pixel coordinates.

```bash
python export_onnx.py --dataset WFLW --output face_landmarker.onnx
```

## Performance

|                             | WFLW           | 300W           | COFW           | Training                     | Inference                   |
| --------------------------- | -------------- | -------------- | -------------- | ---------------------------- | --------------------------- |
| Quantized Heatmap Generator | L2 Loss: 26.72 | L2 Loss: 14.12 | L2 Loss: 30.32 | train_heatmap_generator.py   | test_heatmap_generator.py   |
| ORFormer                    | L2 Loss: 20.22 | L2 Loss: 10.97 | L2 Loss: 23.06 | train_ORFormer.py            | test_ORFormer.py            |
| Integration                 | NME Loss: 3.86 | NME Loss: 2.90 | NME Loss: 4.46 | train_HGNet_with_ORFormer.py | test_HGNet_with_ORFormer.py |

> [!NOTE]
> In the released `weights.zip`, `weights/VQVAE/300W/best_model.pt` is byte-identical to `weights/ORFormer/300W/best_model.pt`, so the 300W heatmap generator scores L2 15.32 instead of 14.12. All other numbers reproduce. `train_ORFormer.py --dataset 300W` also starts from this file.

## Links

[Datasets](https://drive.google.com/file/d/1JmKIstIrfp0mtfTvM0cxMHyZVxYB3k_0/view?usp=share_link)

[weights](https://drive.google.com/file/d/1ebTTbJb9Hsp2bLiUsFgm7_PvtdtsPgPH/view?usp=share_link)

## Citation

```plain
@InProceedings{Chiang_2025_WACV,
    author    = {Chiang, Jui-Che and Hu, Hou-Ning and Hou, Bo-Syuan and Tseng, Chia-Yu and Liu, Yu-Lun and Chen, Min-Hung and Lin, Yen-Yu},
    title     = {ORFormer: Occlusion-Robust Transformer for Accurate Facial Landmark Detection},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {784-793}
}
```

## Acknowledgement

This repository is build on top of [SLPT](https://github.com/Jiahao-UTS/SLPT-master/tree/main), [ADNet](https://github.com/huangyangyu/ADNet/tree/main), [STAR](https://github.com/ZhenglinZhou/STAR/tree/master) and [vit_pytorch](https://github.com/lucidrains/vit-pytorch). Huge appreciation for the excellent codebases provided by these projects!
