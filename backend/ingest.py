"""Ingest the PDF documents of data/ into Qdrant.

For every PDF, the pipeline:
  1. extracts the text, tables and images of each page (PyMuPDF);
  2. asks the vision-language model to describe each image (Ollama);
  3. splits the text into overlapping chunks of tokens;
  4. embeds text (BGE-M3) and images (CLIP) and stores them in Qdrant.
"""

import io
import uuid

import ollama
import pymupdf
from PIL import Image
from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer

import config
from embeddings import clip_image_model, text_model


# 1. Extraction ----------------------------------------------------------------

def clean_text(text: str) -> str:
    """Drop letter-spaced banners (e.g. "L A G R A N D E") and collapse whitespace."""
    kept = []
    for line in text.splitlines():
        words = line.split()
        if len(words) >= 5 and all(len(word) == 1 for word in words):
            continue
        kept.append(line)
    return " ".join(" ".join(kept).split())


def save_image(doc: pymupdf.Document, xref: int, path, max_side: int = 1024) -> None:
    """Save an embedded image as an RGB JPEG no larger than max_side pixels."""
    pix = pymupdf.Pixmap(doc, xref)
    if pix.n - pix.alpha >= 4:  # CMYK images are converted to RGB
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    image.thumbnail((max_side, max_side))
    image.save(path, "JPEG", quality=90)


def extract_pdf(pdf_path) -> tuple[list[dict], list[dict], list[dict]]:
    """Return the text of each page, its tables and the images worth keeping.

    Tables are kept as Markdown so that their rows and columns survive, and their
    area is left out of the page text so that their content is not stored twice.
    """
    doc = pymupdf.open(pdf_path)
    pages, tables, images, seen = [], [], [], set()
    for page in doc:
        number = page.number + 1

        found = page.find_tables().tables
        boxes = [pymupdf.Rect(table.bbox) for table in found]
        for table in found:
            tables.append({"page": number, "markdown": table.to_markdown()})

        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        text = "\n".join(
            block[4] for block in blocks
            if block[6] == 0 and not any(pymupdf.Rect(block[:4]).intersects(box) for box in boxes)
        )
        pages.append({"page": number, "text": clean_text(text)})

        for img in page.get_images(full=True):
            xref, width, height = img[0], img[2], img[3]
            if xref in seen or min(width, height) < config.MIN_IMAGE_SIZE:
                continue
            seen.add(xref)
            path = config.IMAGES_DIR / f"{pdf_path.stem}_p{number}_{xref}.jpg"
            save_image(doc, xref, path)
            images.append({"page": number, "path": path})
    return pages, tables, images


# 2. Image descriptions --------------------------------------------------------

CAPTION_PROMPT = (
    "Décris cette image extraite de la brochure de l'EMSI en 2 ou 3 phrases. "
    "Recopie le texte visible s'il y en a."
)


def describe_image(path, attempts: int = 3) -> str:
    """Ask the vision-language model for a short French description of an image.

    Small models occasionally return an empty answer, so the request is retried.
    """
    for _ in range(attempts):
        response = ollama.chat(
            model=config.VLM_MODEL,
            messages=[{"role": "user", "content": CAPTION_PROMPT, "images": [str(path)]}],
            think=False,
        )
        caption = response.message.content.strip()
        if caption:
            return caption
    return "Image extraite de la brochure de l'EMSI."


# 3. Chunking ------------------------------------------------------------------

def split_text(text: str, tokenizer) -> list[str]:
    """Split a text into chunks of CHUNK_TOKENS tokens sharing CHUNK_OVERLAP tokens."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    step = config.CHUNK_TOKENS - config.CHUNK_OVERLAP
    chunks = []
    for start in range(0, len(ids), step):
        chunks.append(tokenizer.decode(ids[start:start + config.CHUNK_TOKENS]))
        if start + config.CHUNK_TOKENS >= len(ids):
            break
    return chunks


# 4. Embedding and storage -----------------------------------------------------

def create_collection(client: QdrantClient) -> None:
    """(Re)create the collection with one named vector per modality."""
    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)
    client.create_collection(
        collection_name=config.COLLECTION,
        vectors_config={
            "text": models.VectorParams(size=config.TEXT_DIM, distance=models.Distance.COSINE),
            "image": models.VectorParams(size=config.IMAGE_DIM, distance=models.Distance.COSINE),
        },
    )
    client.create_payload_index(
        config.COLLECTION, field_name="page", field_schema=models.PayloadSchemaType.INTEGER
    )


def run(log=print) -> int:
    """Rebuild the collection from every PDF in DATA_DIR and return the number of points.

    `log` receives progress messages: print in a terminal, st.write in Streamlit.
    """
    pdfs = sorted(config.DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF found in {config.DATA_DIR}")
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Steps 1 and 2: extract each PDF, then describe its images with the VLM
    documents = []
    for pdf in pdfs:
        pages, tables, images = extract_pdf(pdf)
        log(f"{pdf.name}: {len(pages)} pages, {len(tables)} tables, {len(images)} images")
        for i, image in enumerate(images, start=1):
            image["caption"] = describe_image(image["path"])
            log(f"  described image {i}/{len(images)} (page {image['page']})")
        documents.append((pdf, pages, tables, images))
    ollama.generate(model=config.VLM_MODEL, keep_alive=0)  # free GPU memory for the embedders

    # Step 3: one record per text chunk, per table and per image
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL)
    records = []
    for pdf, pages, tables, images in documents:
        for page in pages:
            for chunk in split_text(page["text"], tokenizer):
                if len(chunk.split()) >= config.MIN_CHUNK_WORDS:
                    records.append({"type": "text", "content": chunk,
                                    "page": page["page"], "source": pdf.name})
        for table in tables:
            records.append({"type": "table", "content": table["markdown"],
                            "page": table["page"], "source": pdf.name})
        for image in images:
            records.append({"type": "image", "content": image["caption"], "page": image["page"],
                            "source": pdf.name, "image": image["path"].name})
    log(f"{len(records)} records to embed")

    # Step 4: embed and store
    # Every record gets a "text" vector; images also get an "image" vector
    text_vectors = text_model().encode(
        [record["content"] for record in records],
        batch_size=16, normalize_embeddings=True, show_progress_bar=True,
    )
    image_records = [record for record in records if record["type"] == "image"]
    image_vectors = clip_image_model().encode(
        [Image.open(config.IMAGES_DIR / record["image"]) for record in image_records],
        batch_size=16, show_progress_bar=True,
    ) if image_records else []
    clip_by_image = {record["image"]: v for record, v in zip(image_records, image_vectors)}

    points = []
    for record, text_vector in zip(records, text_vectors):
        vector = {"text": text_vector.tolist()}
        if record["type"] == "image":
            vector["image"] = clip_by_image[record["image"]].tolist()
        points.append(models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload=record))

    client = QdrantClient(url=config.QDRANT_URL)
    create_collection(client)
    client.upload_points(config.COLLECTION, points=points, batch_size=64)
    log(f"Stored {len(points)} points in collection '{config.COLLECTION}'")
    return len(points)


if __name__ == "__main__":
    run()
