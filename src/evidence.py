from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class EvidenceItem:
    index: int
    similarity: float
    distance: float

class EvidenceIndex:
    def __init__(self, embeddings: np.ndarray, require_faiss: bool = False):
        self.embeddings = np.asarray(embeddings, dtype="float32")
        self.index = None
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)
        except Exception as e:
            if require_faiss:
                raise RuntimeError("Research Mode requires faiss-cpu, but FAISS could not be imported.") from e
            self.index = None

    def search(self, query_vector: np.ndarray, k: int = 5, exclude_index: int | None = None) -> list[EvidenceItem]:
        q = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        wanted = min(len(self.embeddings), k + (1 if exclude_index is not None else 0))
        if self.index is not None:
            sims, idxs = self.index.search(q, wanted)
            pairs = zip(idxs[0].tolist(), sims[0].tolist())
        else:
            sims = self.embeddings @ q[0]
            idxs = np.argsort(-sims)[:wanted]
            pairs = [(int(i), float(sims[i])) for i in idxs]
        out = []
        for idx, sim in pairs:
            if idx < 0 or idx == exclude_index:
                continue
            sim = max(-1.0, min(1.0, float(sim)))
            # cosine distance on normalized vectors
            out.append(EvidenceItem(idx, sim, 1.0 - sim))
            if len(out) >= k:
                break
        return out
