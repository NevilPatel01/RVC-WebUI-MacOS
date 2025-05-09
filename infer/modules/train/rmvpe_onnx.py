import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

from infer.modules.vc.rmvpe import RMVPE


def convert_to_onnx():
    model_path = "assets/rmvpe/rmvpe.pt"
    onnx_path = "assets/rmvpe/rmvpe.onnx"

    print("Converting RMVPE model to ONNX format...")
    try:
        RMVPE.convert_to_onnx(model_path, onnx_path)
        print(f"ONNX model saved to: {onnx_path}")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    convert_to_onnx()
