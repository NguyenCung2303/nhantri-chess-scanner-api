import json
import shutil
from pathlib import Path

import cv2
import fitz


START_PAGE = 3
ZOOM = 4.0
MIN_BOARD_AREA = 80000
FILES = "abcdefgh"


def ensure_dirs(work_dir: Path):
    page_dir = work_dir / "pages"
    board_dir = work_dir / "boards"
    square_dir = work_dir / "squares"

    page_dir.mkdir(parents=True, exist_ok=True)
    board_dir.mkdir(parents=True, exist_ok=True)
    square_dir.mkdir(parents=True, exist_ok=True)

    return page_dir, board_dir, square_dir


def render_pdf_to_images(pdf_path: Path, page_dir: Path):
    doc = fitz.open(pdf_path)
    page_paths = []

    for page_index in range(START_PAGE - 1, len(doc)):
        page = doc[page_index]
        matrix = fitz.Matrix(ZOOM, ZOOM)
        pix = page.get_pixmap(matrix=matrix)

        page_number = page_index + 1
        output_path = page_dir / f"page_{page_number}.png"
        pix.save(output_path)

        page_paths.append((page_number, output_path))

    return page_paths


def split_board_to_squares(board_image_path: Path, board_key: str, square_dir: Path):
    board = cv2.imread(str(board_image_path))
    if board is None:
        raise ValueError(f"Cannot read board image: {board_image_path}")

    height, width = board.shape[:2]
    size = min(width, height)

    x0 = max((width - size) // 2, 0)
    y0 = max((height - size) // 2, 0)
    board = board[y0:y0 + size, x0:x0 + size]

    square_size = size // 8
    board_square_dir = square_dir / board_key
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

    return board_square_dir


def detect_side_to_move(page_image, x, y, w, h):
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


def detect_chess_boards(page_image_path: Path, page_number: int, board_dir: Path, square_dir: Path):
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
        cv2.CHAIN_APPROX_SIMPLE
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

    items = []

    for index, (x, y, w, h) in enumerate(candidates, start=1):
        side_to_move, side_confidence = detect_side_to_move(original, x, y, w, h)

        padding = 8

        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, original.shape[1])
        y2 = min(y + h + padding, original.shape[0])

        crop = original[y1:y2, x1:x2]

        board_key = f"page_{page_number}_board_{index}"
        board_path = board_dir / f"{board_key}.png"
        cv2.imwrite(str(board_path), crop)

        board_square_dir = split_board_to_squares(
            board_image_path=board_path,
            board_key=board_key,
            square_dir=square_dir
        )

        items.append({
            "boardKey": board_key,
            "page": page_number,
            "boardIndex": index,
            "boardImagePath": str(board_path),
            "squaresDir": str(board_square_dir),
            "sideToMove": side_to_move,
            "sideConfidence": side_confidence
        })

    return items


def scan_pdf_to_boards(pdf_path: Path, work_dir: Path):
    if work_dir.exists():
        shutil.rmtree(work_dir)

    page_dir, board_dir, square_dir = ensure_dirs(work_dir)

    page_paths = render_pdf_to_images(
        pdf_path=pdf_path,
        page_dir=page_dir
    )

    all_items = []
    summary = []

    for page_number, page_path in page_paths:
        items = detect_chess_boards(
            page_image_path=page_path,
            page_number=page_number,
            board_dir=board_dir,
            square_dir=square_dir
        )

        all_items.extend(items)

        summary.append({
            "page": page_number,
            "boards": len(items)
        })

    metadata_path = work_dir / "board_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)

    summary_path = work_dir / "scan_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return all_items