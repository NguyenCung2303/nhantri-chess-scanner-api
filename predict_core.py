import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "piece_classifier.pth"
CLASS_PATH = BASE_DIR / "models" / "classes.json"

IMAGE_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PieceCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


_piece_model = None
_piece_classes = None


def load_model_once():
    global _piece_model
    global _piece_classes

    if _piece_model is not None and _piece_classes is not None:
        return _piece_model, _piece_classes

    with CLASS_PATH.open("r", encoding="utf-8") as f:
        classes = json.load(f)

    model = PieceCNN(num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )
    model.eval()

    _piece_model = model
    _piece_classes = classes

    return model, classes


def predict_square(model, classes, image_path: Path):
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    return classes[pred_idx.item()], conf.item()


def labels_to_fen(labels, side_to_move="w"):
    files = "abcdefgh"
    ranks = range(8, 0, -1)

    if side_to_move not in {"w", "b"}:
        side_to_move = "w"

    piece_to_fen = {
        "wp": "P",
        "wn": "N",
        "wb": "B",
        "wr": "R",
        "wq": "Q",
        "wk": "K",
        "bp": "p",
        "bn": "n",
        "bb": "b",
        "br": "r",
        "bq": "q",
        "bk": "k",
    }

    fen_rows = []

    for rank in ranks:
        empty_count = 0
        row_text = ""

        for file_name in files:
            square = f"{file_name}{rank}"
            label = labels.get(square, "empty")

            if label == "empty":
                empty_count += 1
            else:
                if empty_count > 0:
                    row_text += str(empty_count)
                    empty_count = 0

                row_text += piece_to_fen[label]

        if empty_count > 0:
            row_text += str(empty_count)

        fen_rows.append(row_text)

    return "/".join(fen_rows) + f" {side_to_move} - - 0 1"


def predict_board_item(board_item, confidence_threshold=0.55):
    model, classes = load_model_once()

    squares_dir = Path(board_item["squaresDir"])

    board_labels = {}
    simple_labels = {}
    confidences = []

    for square_path in sorted(squares_dir.glob("*.png")):
        square = square_path.stem

        label, conf = predict_square(
            model=model,
            classes=classes,
            image_path=square_path
        )

        final_label = label

        if conf < confidence_threshold:
            final_label = "unknown"

        board_labels[square] = {
            "label": final_label,
            "rawLabel": label,
            "confidence": round(conf, 4)
        }

        simple_labels[square] = final_label
        confidences.append(conf)

    has_unknown = "unknown" in simple_labels.values()

    fen = None
    if not has_unknown:
        fen = labels_to_fen(
            labels=simple_labels,
            side_to_move=board_item.get("sideToMove", "w")
        )

    avg_confidence = 0.0
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)

    return {
        **board_item,
        "fen": fen,
        "hasUnknown": has_unknown,
        "avgConfidence": round(avg_confidence, 4),
        "labels": board_labels
    }


def predict_boards(board_items, confidence_threshold=0.55):
    results = []

    for item in board_items:
        results.append(
            predict_board_item(
                board_item=item,
                confidence_threshold=confidence_threshold
            )
        )

    return results