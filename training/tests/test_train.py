import torch
import types
import tempfile
from training.train import freeze_and_unfreeze_layers, train

class DummyModule:
    def __init__(self):
        self.parameters_called = False
    def parameters(self):
        self.parameters_called = True
        param = torch.nn.Parameter(torch.randn(2, 2))
        param.requires_grad = False
        return [param]

def test_freeze_and_unfreeze_layers_sets_requires_grad():
    class DummyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.randn(1))
        def parameters(self):
            return [self.param]

    class DummyEncoder:
        def __init__(self):
            self.named_modules_called = False
        def named_modules(self):
            return [("layer1", DummyLayer()) for _ in range(12)]

    class DummyDecoderModel:
        def __init__(self):
            self.decoder = types.SimpleNamespace(
                layers=[DummyLayer() for _ in range(4)],
                embed_positions=DummyLayer(),
                embed_tokens=DummyLayer()
            )

    class DummyDecoder:
        def __init__(self):
            self.model = DummyDecoderModel()

    class DummyModel:
        def __init__(self):
            self.encoder = DummyEncoder()
            self.decoder = DummyDecoder()
        def parameters(self):
            return [p for p in
                    [module.param for name, module in self.encoder.named_modules()] +
                    [layer.param for layer in self.decoder.model.decoder.layers] +
                    list(self.decoder.model.decoder.embed_positions.parameters()) +
                    list(self.decoder.model.decoder.embed_tokens.parameters())]

    model = DummyModel()
    freeze_and_unfreeze_layers(model)

    encoder_layers = list(model.encoder.named_modules())[-4:]
    for name, module in encoder_layers:
        for param in module.parameters():
            assert param.requires_grad

def test_train_runs_one_epoch(monkeypatch):
    class DummyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.randn(1))
        def parameters(self):
            return [self.param]

    class DummyEncoder:
        def named_modules(self):
            return [(f"layer{i}", DummyLayer()) for i in range(12)]

    class DummyDecoderModel:
        def __init__(self):
            self.decoder = types.SimpleNamespace(
                layers=[DummyLayer() for _ in range(4)],
                embed_positions=DummyLayer(),
                embed_tokens=DummyLayer()
            )

    class DummyDecoder:
        def __init__(self):
            self.model = DummyDecoderModel()

    class DummyProcessor:
        tokenizer = types.SimpleNamespace(pad_token_id=0)
        def batch_decode(self, *args, **kwargs):
            return ["decoded"] * 4
        def __call__(self, image, return_tensors=None):
            return types.SimpleNamespace(pixel_values=torch.rand(1, 3, 64, 64))
        def save_pretrained(self, path):
            pass  # dummy method to avoid AttributeError

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = DummyEncoder()
            self.decoder = DummyDecoder()
            self.training = False
        def train(self):
            self.training = True
        def eval(self):
            self.training = False
        def to(self, device):
            return self
        def parameters(self):
            encoder_params = []
            for name, module in self.encoder.named_modules():
                encoder_params.extend(list(module.parameters()))
            decoder_layers = self.decoder.model.decoder.layers
            decoder_params = []
            for layer in decoder_layers:
                decoder_params.extend(list(layer.parameters()))
            decoder_params.extend(list(self.decoder.model.decoder.embed_positions.parameters()))
            decoder_params.extend(list(self.decoder.model.decoder.embed_tokens.parameters()))
            return encoder_params + decoder_params
        def __call__(self, pixel_values=None, labels=None):
            device = labels.device if labels is not None else torch.device("cpu")
            logits = torch.randn(labels.size(0), labels.size(1), 10, device=device)
            loss = torch.tensor(1.0, requires_grad=True, device=device)
            return types.SimpleNamespace(loss=loss, logits=logits)
        def save_pretrained(self, path):
            pass

    dummy_batch = {
        "pixel_values": torch.rand(2, 3, 64, 64),  # batch_size=2, channels=3, 64x64 images
        "labels": torch.randint(0, 10, (2, 5))     # batch_size=2, seq_len=5, vocab_size=10 (example)
    }

    monkeypatch.setattr("training.train.DonutProcessor.from_pretrained", lambda model_name: DummyProcessor())
    monkeypatch.setattr("training.train.VisionEncoderDecoderModel.from_pretrained", lambda model_name: DummyModel())
    monkeypatch.setattr("training.train.IDDataset", lambda dir, processor: [dummy_batch] * 2)
    monkeypatch.setattr("training.train.DataLoader", lambda dataset, batch_size, shuffle, collate_fn: dataset)
    monkeypatch.setattr("training.train.wandb.init", lambda project: None)
    monkeypatch.setattr("training.train.wandb.log", lambda data: None)
    monkeypatch.setattr("training.train.os.makedirs", lambda path, exist_ok=True: None)
    monkeypatch.setattr("training.train.tqdm", lambda iterable, desc=None: iterable)

    monkeypatch.setattr("training.train.TrainingConfig", types.SimpleNamespace(
        MODEL_NAME="dummy",
        TRAIN_DIR="dummy",
        VAL_DIR="dummy",
        BATCH_SIZE=2,
        LEARNING_RATE=1e-4,
        MAX_EPOCHS=1,
        CONFIDENCE_THRESHOLD=0.8,
        CHECKPOINT_DIR=tempfile.mkdtemp()
    ))

    train()
