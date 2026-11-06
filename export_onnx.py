"""HGNet + ORFormer 모델을 ONNX로 변환한다.

입력: img256 (1, 3, 256, 256), img64 (1, 3, 64, 64). 같은 얼굴 크롭을 RGB,
ImageNet 정규화한 텐서다(img64는 img256을 64x64로 줄인 것).
출력: landmarks (1, N, 2). 256x256 크롭 기준 (x, y) 픽셀 좌표다.

Examples:
    python export_onnx.py --dataset WFLW --output face_landmarker.onnx
"""

import argparse

import numpy as np
import onnxruntime
import torch
from torch import nn

from Config import cfg
from Dataloader import denorm_points
from Model.simple_vit import ORFormer
from Model.StackedHGNet import IntergrationStackedHGNet
from Model.VQVAE import VQVAE

DATASET_CFG = {"WFLW": cfg.WFLW, "300W": cfg.W300, "COFW": cfg.COFW}


class FaceLandmarker(nn.Module):
    """test_HGNet_with_ORFormer.py의 추론 과정을 모듈 하나로 묶는다."""

    def __init__(self, dataset: str, nstack: int = 4) -> None:
        super().__init__()
        c = DATASET_CFG[dataset]
        self.hgnet = IntergrationStackedHGNet(
            classes_num=[c.NUM_POINT, c.NUM_EDGE, c.NUM_POINT],
            edge_info=c.EDGE_INFO,
            nstack=nstack,
        )
        self.hgnet.load_weights(f"weights/HGNet/{dataset}/best_model.pt")
        vit = ORFormer(
            image_size=16,
            patch_size=1,
            num_classes=2048,
            dim=256,
            depth=3,
            heads=8,
            mlp_dim=512,
            channels=256,
        )
        self.vqvae = VQVAE(
            h_dim=128,
            res_h_dim=32,
            output_dim=c.NUM_EDGE,
            n_res_layers=2,
            n_embeddings=2048,
            embedding_dim=256,
            code_dim=256,
            beta=0.25,
            save_img_embedding_map=False,
            vit=vit,
        )
        self.vqvae.load_weights(f"weights/ORFormer/{dataset}/best_model.pt")

    def forward(self, img256: torch.Tensor, img64: torch.Tensor) -> torch.Tensor:
        reference_heatmaps = self.vqvae(img64)[1]
        _, landmarks = self.hgnet(img256, reference_heatmaps=reference_heatmaps)
        # 64x64 히트맵 좌표 -> 256x256 크롭 좌표
        return denorm_points(landmarks, 64, 64) * 4


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HGNet + ORFormer to ONNX")
    parser.add_argument("--dataset", choices=list(DATASET_CFG), default="WFLW")
    parser.add_argument("--nstack", type=int, default=4)
    parser.add_argument("--output", default="face_landmarker.onnx")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    model = FaceLandmarker(args.dataset, args.nstack).eval()
    inputs = (torch.randn(1, 3, 256, 256), torch.randn(1, 3, 64, 64))
    torch.onnx.export(
        model,
        inputs,
        args.output,
        input_names=["img256", "img64"],
        output_names=["landmarks"],
        opset_version=args.opset,
        dynamo=False,
    )

    # 변환 결과가 PyTorch 출력과 같은지 확인한다.
    with torch.no_grad():
        expected = model(*inputs).numpy()
    session = onnxruntime.InferenceSession(
        args.output, providers=["CPUExecutionProvider"]
    )
    actual = session.run(
        None, {"img256": inputs[0].numpy(), "img64": inputs[1].numpy()}
    )[0]
    max_diff = float(np.abs(actual - expected).max())
    assert max_diff < 1e-2, (
        f"ONNX 출력이 PyTorch와 다릅니다 (최대 차이 {max_diff:.4g}px)"
    )
    print(f"Saved {args.output} (max diff vs PyTorch: {max_diff:.2e}px)")


if __name__ == "__main__":
    main()
