from __future__ import annotations
import json
import math
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from openai import OpenAI
from qdrant_client import QdrantClient, models

DATA_DIR = Path(__file__).resolve().parent / "data"
COLLECTION_NAME = "ratings"


def _load_config() -> Dict:
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    if cfg_path.exists():
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return {}


config = _load_config()
MODEL_NAME = config.get("embedding_model", "text-embedding-qwen3-embedding-4b")

openai_client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
qdrant = QdrantClient(path=str(DATA_DIR / "qdrant"))


def get_embedding(text: str, model: Optional[str] = None) -> List[float]:
    model = model or MODEL_NAME
    resp = openai_client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding


def init_db(force_reset: bool = False) -> None:
    """Create Qdrant collection if it doesn't exist or force recreate."""
    DATA_DIR.mkdir(exist_ok=True)
    dim = len(get_embedding("test"))
    vectors_config = models.VectorParams(size=dim, distance=models.Distance.COSINE)
    if force_reset:
        qdrant.recreate_collection(COLLECTION_NAME, vectors_config=vectors_config)
    else:
        try:
            qdrant.get_collection(COLLECTION_NAME)
        except Exception:
            qdrant.recreate_collection(COLLECTION_NAME, vectors_config=vectors_config)


def add_record(text: str, rating: float) -> str:
    vid = str(uuid.uuid4())
    vector = get_embedding(text)
    point = models.PointStruct(id=vid, vector=vector, payload={"text": text, "rating": float(rating)})
    qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])
    return vid


def update_record(record_id: str, text: Optional[str] = None, rating: Optional[float] = None) -> None:
    rec = get_record(record_id)
    if not rec:
        return
    new_text = text if text is not None else rec["text"]
    new_rating = float(rating) if rating is not None else rec["rating"]
    vector = get_embedding(new_text) if text is not None else rec.get("vector")
    point = models.PointStruct(id=record_id, vector=vector, payload={"text": new_text, "rating": new_rating})
    qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])


def delete_record(record_id: str) -> None:
    qdrant.delete(collection_name=COLLECTION_NAME, points_selector=models.PointIdsList(points=[record_id]))


def get_all_records() -> List[Dict]:
    records: List[Dict] = []
    next_offset = None
    while True:
        res, next_offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=True,
            with_vectors=False,
            limit=100,
            offset=next_offset,
        )
        for p in res:
            records.append({"id": p.id, "text": p.payload.get("text"), "rating": p.payload.get("rating")})
        if next_offset is None:
            break
    return records


def get_all_ids() -> List[str]:
    return [rec["id"] for rec in get_all_records()]


def get_record(record_id: str) -> Optional[Dict]:
    res = qdrant.retrieve(
        collection_name=COLLECTION_NAME, ids=[record_id], with_payload=True, with_vectors=True
    )
    if res:
        p = res[0]
        return {
            "id": p.id,
            "text": p.payload.get("text"),
            "rating": p.payload.get("rating"),
            "vector": p.vector,
        }
    return None


def import_json(path: Path) -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    points = []
    for item in data:
        text = item["text"]
        rating = float(item["rating"])
        vid = str(uuid.uuid4())
        vector = get_embedding(text)
        points.append(
            models.PointStruct(id=vid, vector=vector, payload={"text": text, "rating": rating})
        )
    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def export_json(path: Path) -> None:
    records = get_all_records()
    Path(path).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def recalculate_embeddings(model_name: Optional[str] = None) -> None:
    model_name = model_name or MODEL_NAME
    records = get_all_records()
    points = []
    for rec in records:
        vector = get_embedding(rec["text"], model_name)
        points.append(
            models.PointStruct(
                id=rec["id"], vector=vector, payload={"text": rec["text"], "rating": rec["rating"]}
            )
        )
    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def predict_rating(text: str, n: int, epsilon: float, model_name: Optional[str] = None) -> float:
    vector = get_embedding(text, model_name)
    results = qdrant.search(collection_name=COLLECTION_NAME, query_vector=vector, limit=n, with_payload=True)
    if not results:
        return 0.0
    weights = []
    ratings = []
    for r in results:
        dist = r.score  # cosine distance: smaller value means higher similarity
        weight = 1.0 / (dist + epsilon)
        weights.append(weight)
        ratings.append(r.payload.get("rating", 0.0))
    weights_arr = np.array(weights)
    ratings_arr = np.array(ratings)
    return float(np.sum(weights_arr * ratings_arr) / np.sum(weights_arr))


def grid_search_n_epsilon(
    n_values: List[int], epsilon_values: List[float]
) -> Tuple[int, float, float]:
    points = []
    next_offset = None
    while True:
        res, next_offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=True,
            with_vectors=True,
            limit=100,
            offset=next_offset,
        )
        points.extend(res)
        if next_offset is None:
            break
    best_rmse = float("inf")
    best_n = 0
    best_eps = 0.0
    for n in n_values:
        for eps in epsilon_values:
            errors: List[float] = []
            for p in points:
                search = qdrant.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=p.vector,
                    limit=n + 1,
                    with_payload=True,
                )
                neighbors = [s for s in search if s.id != p.id][:n]
                if not neighbors:
                    continue
                weights = []
                ratings = []
                for nb in neighbors:
                    dist = nb.score  # cosine distance: smaller value means higher similarity
                    weight = 1.0 / (dist + eps)
                    weights.append(weight)
                    ratings.append(nb.payload.get("rating", 0.0))
                weights_arr = np.array(weights)
                ratings_arr = np.array(ratings)
                pred = float(np.sum(weights_arr * ratings_arr) / np.sum(weights_arr))
                errors.append((pred - p.payload.get("rating", 0.0)) ** 2)
            if errors:
                rmse = math.sqrt(float(np.mean(errors)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_n = n
                    best_eps = eps
    return best_n, best_eps, best_rmse
