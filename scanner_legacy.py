import argparse
import json
import shutil
from pathlib import Path

import cv2
import fitz


INPUT_PDF = Path("input/sample.pdf")
OUTPUT_DIR = Path("output")
PAGE_IMAGE_DIR = OUTPUT_DIR / "pages"
BOARD_IMAGE_DIR = OUTPUT_DIR / "boards"
SQUARE_IMAGE_DIR = OUTPUT_DIR / "squares"
BOARD_METADATA_PATH = OUTPUT_DIR / "board_metadata.json"

START_PAGE = 3
ZOOM = 4.0
MIN_BOARD_AREA = 80000

FILES = "abcdefgh"
VALID_LABELS = {
    "empty",
    "wp", "wn", "wb", "wr", "wq", "wk",
    "bp", "bn", "bb", "br", "bq", "bk",
}

PIECE_TO_FEN = {
    "wp": "P", "wn": "N", "wb": "B", "wr": "R", "wq": "Q", "wk": "K",
    "bp": "p", "bn": "n", "bb": "b", "br": "r", "bq": "q", "bk": "k",
}


def ensure_dirs():
    PAGE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    BOARD_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    SQUARE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def clean_output():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    ensure_dirs()


def render_pdf_to_images(pdf_path: Path, start_page: int = START_PAGE, zoom: float = ZOOM):
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    ensure_dirs()
    doc = fitz.open(pdf_path)
    page_paths = []

    for page_index in range(start_page - 1, len(doc)):
        page = doc[page_index]
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)

        real_page_number = page_index + 1
        output_path = PAGE_IMAGE_DIR / f"page_{real_page_number}.png"
        pix.save(output_path)

        page_paths.append((real_page_number, output_path))

    return page_paths



def detect_side_to_move(page_image, x, y, w, h):
    """
    Detect side to move by checking for a black dot near the top-right side
    outside the chess board.

    black dot found => "b"
    otherwise       => "w"
    """
    image_height, image_width = page_image.shape[:2]

    search_x1 = min(x + w, image_width)
    search_x2 = min(x + w + int(w * 0.28), image_width)

    search_y1 = max(y - int(h * 0.08), 0)
    search_y2 = min(y + int(h * 0.28), image_height)

    region = page_image[search_y1:search_y2, search_x1:search_x2]

    if region.size == 0:
        return "w", 0.0

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(
        gray,
        80,
        255,
        cv2.THRESH_BINARY_INV
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_score = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        bx, by, bw, bh = cv2.boundingRect(contour)

        if bh == 0:
            continue

        ratio = bw / float(bh)

        is_size_ok = 20 <= area <= 2500
        is_round_like = 0.55 <= ratio <= 1.45

        if not (is_size_ok and is_round_like):
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * 3.14159265 * area / (perimeter * perimeter)

        if circularity < 0.35:
            continue

        best_score = max(best_score, min(1.0, circularity))

    if best_score > 0:
        return "b", round(float(best_score), 4)

    return "w", 0.0


def detect_chess_boards(page_image_path: Path, page_number: int):
    image = cv2.imread(str(page_image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {page_image_path}")

    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        ratio = w / float(h)

        is_square_like = 0.85 <= ratio <= 1.15
        is_big_enough = area > MIN_BOARD_AREA

        if is_square_like and is_big_enough:
            candidates.append((x, y, w, h))

    candidates = sorted(candidates, key=lambda box: (box[1], box[0]))
    saved_boards = []
    board_metadata = []

    for index, (x, y, w, h) in enumerate(candidates, start=1):
        side_to_move, side_confidence = detect_side_to_move(original, x, y, w, h)
        padding = 8

        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, original.shape[1])
        y2 = min(y + h + padding, original.shape[0])

        crop = original[y1:y2, x1:x2]
        board_key = f"page_{page_number}_board_{index}"
        board_path = BOARD_IMAGE_DIR / f"{board_key}.png"
        cv2.imwrite(str(board_path), crop)

        split_board_to_squares(board_path, board_key)
        saved_boards.append(board_path)

        board_metadata.append({
            "board_key": board_key,
            "page": page_number,
            "board_index": index,
            "board_image": str(board_path),
            "side_to_move": side_to_move,
            "side_confidence": side_confidence,
            "bbox": {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
            }
        })

    return saved_boards, board_metadata


def split_board_to_squares(board_image_path: Path, board_key: str):
    board = cv2.imread(str(board_image_path))
    if board is None:
        raise ValueError(f"Cannot read board image: {board_image_path}")

    height, width = board.shape[:2]
    size = min(width, height)

    # Cắt giữa ảnh để tránh bị lệch nếu crop có padding không đều.
    x0 = max((width - size) // 2, 0)
    y0 = max((height - size) // 2, 0)
    board = board[y0:y0 + size, x0:x0 + size]

    square_size = size // 8
    board_square_dir = SQUARE_IMAGE_DIR / board_key
    board_square_dir.mkdir(parents=True, exist_ok=True)

    for row in range(8):
        for col in range(8):
            x1 = col * square_size
            y1 = row * square_size
            x2 = (col + 1) * square_size
            y2 = (row + 1) * square_size

            square = board[y1:y2, x1:x2]
            square_name = f"{FILES[col]}{8 - row}"
            square_path = board_square_dir / f"{square_name}.png"
            cv2.imwrite(str(square_path), square)


def create_labels_template():
    if not SQUARE_IMAGE_DIR.exists():
        raise FileNotFoundError("No output/squares folder found. Run: python3 main.py scan")

    template = {}
    board_dirs = sorted([p for p in SQUARE_IMAGE_DIR.iterdir() if p.is_dir()])

    for board_dir in board_dirs:
        labels = {}
        for rank in range(8, 0, -1):
            for file_name in FILES:
                square = f"{file_name}{rank}"
                labels[square] = "unknown"
        template[board_dir.name] = labels

    output_path = Path("labels_template.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print(f"Created {output_path} with {len(template)} boards.")
    print("Copy it to labels.json, then replace unknown with labels like empty, wp, bk...")



def load_board_metadata():
    if not BOARD_METADATA_PATH.exists():
        return {}

    with BOARD_METADATA_PATH.open("r", encoding="utf-8") as f:
        items = json.load(f)

    return {
        item["board_key"]: item
        for item in items
    }


def labels_to_fen(labels: dict, side_to_move: str = "w"):
    if side_to_move not in {"w", "b"}:
        side_to_move = "w"

    fen_rows = []

    for rank in range(8, 0, -1):
        empty_count = 0
        row_text = ""

        for file_name in FILES:
            square = f"{file_name}{rank}"
            label = labels.get(square, "empty")

            if label == "unknown":
                raise ValueError(f"Square {square} is still unknown")

            if label not in VALID_LABELS:
                raise ValueError(f"Invalid label at {square}: {label}")

            if label == "empty":
                empty_count += 1
            else:
                if empty_count > 0:
                    row_text += str(empty_count)
                    empty_count = 0
                row_text += PIECE_TO_FEN[label]

        if empty_count > 0:
            row_text += str(empty_count)

        fen_rows.append(row_text)

    board_fen = "/".join(fen_rows)
    return f"{board_fen} {side_to_move} - - 0 1"


def generate_fen_from_labels():
    labels_path = Path("labels.json")

    if not labels_path.exists():
        print("No labels.json found. Create one from labels_template.json first.")
        return

    with labels_path.open("r", encoding="utf-8") as f:
        all_labels = json.load(f)

    metadata_by_key = load_board_metadata()

    results = {}
    errors = {}

    for board_key, board_labels in all_labels.items():
        try:
            side_to_move = metadata_by_key.get(board_key, {}).get("side_to_move", "w")
            results[board_key] = labels_to_fen(board_labels, side_to_move)
        except Exception as exc:
            errors[board_key] = str(exc)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fen_output_path = OUTPUT_DIR / "fen_results.json"
    error_output_path = OUTPUT_DIR / "fen_errors.json"

    with fen_output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with error_output_path.open("w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(results)} FEN results -> {fen_output_path}")
    if errors:
        print(f"Skipped {len(errors)} boards with errors -> {error_output_path}")

    for board_key, fen in results.items():
        print(board_key, "=>", fen)


def scan_pdf():
    clean_output()
    print(f"Rendering PDF from page {START_PAGE}...")
    page_paths = render_pdf_to_images(INPUT_PDF)

    total_boards = 0
    summary = []
    all_board_metadata = []

    for page_number, page_path in page_paths:
        boards, board_metadata = detect_chess_boards(page_path, page_number)

        total_boards += len(boards)
        summary.append({"page": page_number, "boards": len(boards)})
        all_board_metadata.extend(board_metadata)

        black_count = sum(1 for item in board_metadata if item["side_to_move"] == "b")
        print(
            f"Page {page_number}: found {len(boards)} boards "
            f"({black_count} black-to-move detected)"
        )

    summary_path = OUTPUT_DIR / "scan_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with BOARD_METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_board_metadata, f, ensure_ascii=False, indent=2)

    print(f"Done. Total boards found: {total_boards}")
    print(f"Boards: {BOARD_IMAGE_DIR}")
    print(f"Squares: {SQUARE_IMAGE_DIR}")
    print(f"Metadata: {BOARD_METADATA_PATH}")
    create_labels_template()


def main():
    parser = argparse.ArgumentParser(description="PDF chess board scanner")
    parser.add_argument(
        "command",
        choices=["scan", "template", "fen", "clean"],
        help="scan: crop boards/squares, template: create labels_template.json, fen: generate FEN from labels.json, clean: remove output",
    )
    args = parser.parse_args()

    if args.command == "scan":
        scan_pdf()
    elif args.command == "template":
        create_labels_template()
    elif args.command == "fen":
        generate_fen_from_labels()
    elif args.command == "clean":
        clean_output()
        print("Output cleaned.")


if __name__ == "__main__":
    main()
