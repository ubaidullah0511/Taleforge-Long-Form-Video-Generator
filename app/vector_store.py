import chromadb
from chromadb.config import Settings

from app.config import settings
from app.embeddings import embed
from app.models import ClipRecord

_client = chromadb.PersistentClient(
    path=str(settings.chroma_dir),
    settings=Settings(anonymized_telemetry=False),
)
_collection = _client.get_or_create_collection(
    name="clips", metadata={"hnsw:space": "cosine"}
)


def _to_metadata(clip: ClipRecord) -> dict:
    data = clip.model_dump(exclude={"id", "keywords"})
    data["keywords"] = ",".join(clip.keywords)
    return data


def _from_metadata(clip_id: str, metadata: dict) -> ClipRecord:
    data = dict(metadata)
    data["keywords"] = [k for k in data.get("keywords", "").split(",") if k]
    return ClipRecord(id=clip_id, **data)


def upsert_clip(clip: ClipRecord) -> None:
    _collection.upsert(
        ids=[clip.id],
        embeddings=[embed(f"{clip.description} {' '.join(clip.keywords)}")],
        metadatas=[_to_metadata(clip)],
        documents=[clip.description],
    )


def query_similar(text: str, top_k: int = 5) -> list[tuple[ClipRecord, float]]:
    if _collection.count() == 0:
        return []
    result = _collection.query(query_embeddings=[embed(text)], n_results=min(top_k, _collection.count()))
    matches = []
    for clip_id, metadata, distance in zip(
        result["ids"][0], result["metadatas"][0], result["distances"][0]
    ):
        similarity = 1 - distance  # cosine space: distance = 1 - cosine_similarity
        matches.append((_from_metadata(clip_id, metadata), similarity))
    return matches


def get_by_source_id(source: str, source_id: str) -> ClipRecord | None:
    if not source_id:
        return None
    result = _collection.get(where={"$and": [{"source": source}, {"source_id": source_id}]})
    if not result["ids"]:
        return None
    return _from_metadata(result["ids"][0], result["metadatas"][0])


def get_by_fingerprint(fingerprint: str) -> ClipRecord | None:
    if not fingerprint:
        return None
    result = _collection.get(where={"fingerprint": fingerprint})
    if not result["ids"]:
        return None
    return _from_metadata(result["ids"][0], result["metadatas"][0])
