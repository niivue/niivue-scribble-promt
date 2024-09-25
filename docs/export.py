import torch
import torch.nn.functional as F
from scribbleprompt.unet import UNet

class Predictor:
    """
    wrapper for ScribblePrompt-UNet model with ONNX export functionality.
    """
    def __init__(self, path: str, verbose: bool = True):
        self.path = path
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model()
        self.load()
        self.model.eval()
        self.to_device()

    def build_model(self):
        """
        build the ScribblePrompt-UNet model.
        """
        self.model = UNet(
            in_channels=5,
            out_channels=1,
            features=[192, 192, 192, 192],
        )

    def load(self):
        """
        load the state of the model from a checkpoint file.
        """
        with open(self.path, "rb") as f:
            state = torch.load(f, map_location=self.device)
            self.model.load_state_dict(state, strict=True)
            if self.verbose:
                print(f"loaded checkpoint from {self.path} to {self.device}")

    def to_device(self):
        """
        move the model to the appropriate device.
        """
        self.model = self.model.to(self.device)

    def export_to_onnx(self, onnx_path="model.onnx"):
        """
        export the model to ONNX format with dynamic H and W (height and width).
        """
        # prepare a dummy input with arbitrary H and W, as ONNX export requires a concrete input shape
        dummy_input = torch.randn(1, 5, 256, 256).to(self.device)
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=20,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            # make H and W dynamic, along with the batch size
            dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                        'output': {0: 'batch_size', 2: 'height', 3: 'width'}}
        )
        print(f"model exported to {onnx_path}")

# usage from CLI
if __name__ == "__main__":
    checkpoint_path = "../checkpoints/ScribblePrompt_unet_v1_nf192_res128.pt"
    predictor = Predictor(checkpoint_path)
    predictor.export_to_onnx("scribbleprompt_unet.onnx")