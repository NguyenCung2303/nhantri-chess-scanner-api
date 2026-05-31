from pathlib import Path
import shutil
import uuid
import json

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse

from scanner_core import scan_pdf_to_boards
from predict_core import predict_boards
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="NhanTri Chess Scanner API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


@app.get("/")
def home():
    return {
        "message": "Chess Scanner API running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/scan-pdf")
async def scan_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {
            "success": False,
            "message": "Only PDF files are allowed"
        }

    import_id = str(uuid.uuid4())

    upload_path = UPLOAD_DIR / f"{import_id}_{file.filename}"
    work_dir = OUTPUT_DIR / import_id
    work_dir.mkdir(parents=True, exist_ok=True)

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    boards = scan_pdf_to_boards(
        pdf_path=upload_path,
        work_dir=work_dir
    )

    items = predict_boards(boards)

    results_path = work_dir / "results.json"

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    fen_count = sum(1 for item in items if item.get("fen"))
    unknown_count = sum(1 for item in items if item.get("hasUnknown"))

    return {
        "success": True,
        "importId": import_id,
        "filename": file.filename,
        "totalBoards": len(boards),
        "predictedBoards": len(items),
        "fenCount": fen_count,
        "unknownCount": unknown_count,
        "itemsUrl": f"/imports/{import_id}/items"
    }


@app.get("/imports/{import_id}/items")
def get_import_items(
        import_id: str,
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0)
):
    results_path = OUTPUT_DIR / import_id / "results.json"

    if not results_path.exists():
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": "Import result not found"
            }
        )

    with results_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    total = len(items)

    paged_items = []

    for item in items[offset:offset + limit]:
        paged_items.append({
            "boardKey": item.get("boardKey"),
            "page": item.get("page"),
            "boardIndex": item.get("boardIndex"),
            "fen": item.get("fen"),
            "sideToMove": item.get("sideToMove"),
            "hasUnknown": item.get("hasUnknown"),
            "avgConfidence": item.get("avgConfidence")
        })

    return {
        "success": True,
        "importId": import_id,
        "total": total,
        "limit": limit,
        "offset": offset,
        "nextOffset": offset + limit if offset + limit < total else None,
        "items": paged_items
    }