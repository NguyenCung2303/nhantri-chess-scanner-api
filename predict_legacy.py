import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


SQUARE_DIR = Path("output/squares")
MODEL_PATH = Path("models/piece_classifier.pth")
CLASS_PATH = Path("models/classes.json")
OUTPUT_LABELS = Path("labels_ai.json")
OUTPUT_FEN = Path("output/fen_ai_results.json")
BOARD_METADATA_PATH = Path("output/board_metadata.json")

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


def load_model():
    with CLASS_PATH.open("r", encoding="utf-8") as f:
        classes = json.load(f)

    model = PieceCNN(num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    return model, classes



def load_board_metadata():
    if not BOARD_METADATA_PATH.exists():
        return {}

    with BOARD_METADATA_PATH.open("r", encoding="utf-8") as f:
        items = json.load(f)

    return {
        item["board_key"]: item
        for item in items
    }


def predict_square(model, classes, image_path):
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

        for file in files:
            square = f"{file}{rank}"
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


def main():
    model, classes = load_model()
    metadata_by_key = load_board_metadata()

    all_labels = {}
    all_fens = {}

    board_dirs = sorted([p for p in SQUARE_DIR.iterdir() if p.is_dir()])

    for board_dir in board_dirs:
        board_key = board_dir.name
        board_labels = {}

        for square_path in sorted(board_dir.glob("*.png")):
            square = square_path.stem
            label, conf = predict_square(model, classes, square_path)

            # Có thể chỉnh ngưỡng này sau
            if conf < 0.55:
                label = "unknown"

            board_labels[square] = {
                "label": label,
                "confidence": round(conf, 4)
            }

        simple_labels = {
            square: data["label"]
            for square, data in board_labels.items()
        }

        all_labels[board_key] = board_labels

        if "unknown" not in simple_labels.values():
            side_to_move = metadata_by_key.get(board_key, {}).get("side_to_move", "w")
            all_fens[board_key] = labels_to_fen(simple_labels, side_to_move)

        print(f"Predicted {board_key}")

    with OUTPUT_LABELS.open("w", encoding="utf-8") as f:
        json.dump(all_labels, f, indent=2, ensure_ascii=False)

    OUTPUT_FEN.parent.mkdir(exist_ok=True)

    with OUTPUT_FEN.open("w", encoding="utf-8") as f:
        json.dump(all_fens, f, indent=2, ensure_ascii=False)

    print("Saved:", OUTPUT_LABELS)
    print("Saved:", OUTPUT_FEN)
    print("Total boards:", len(all_labels))
    print("Boards with FEN:", len(all_fens))
    black_fen_count = sum(1 for fen in all_fens.values() if " b " in fen)
    print("Black-to-move FEN:", black_fen_count)


if __name__ == "__main__":
    main()