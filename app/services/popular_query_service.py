import numpy as np
import logging
import re
from typing import List, Optional, Any

logger = logging.getLogger(__name__)

class PopularQueryService:
    def __init__(self, model_name: str = "all-mpnet-base-v2", similarity_threshold: float = 0.68):
        self.model_name = model_name
        self._model = None
        self.similarity_threshold = similarity_threshold

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip()

    def _embed(self, texts: List[str]) -> np.ndarray:
        self._ensure_model()
        normalized = [self._normalize(t) for t in texts]
        arr = self._model.encode(normalized, convert_to_numpy=True)
        return np.asarray(arr, dtype=float)

    async def save_popular_query(self, question: str, db_manager: Any, similarity_threshold: Optional[float] = None):
        from sentence_transformers import util

        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold

        question_text = question.strip()
        if question_text == "":
            return {"ok": False, "reason": "empty question"}

        try:
            new_emb = self._embed([question_text])[0]
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            async with db_manager.get_connection() as conn:
                await conn.execute(
                    """INSERT INTO "PopularQueries" ("DisplayText", "Count", "LastAsked") VALUES ($1, 1, NOW())""",
                    question_text
                )
            return {"ok": True, "inserted": True, "reason": "embedding failed; inserted without embedding"}

        async with db_manager.get_connection() as conn:
            rows = await conn.fetch(
                """SELECT "Id", "DisplayText", "Count", "Embedding" FROM "PopularQueries" """
            )

            if not rows:
                await conn.execute(
                    """INSERT INTO "PopularQueries" ("DisplayText", "Count", "LastAsked", "Embedding") VALUES ($1, 1, NOW(), $2)""",
                    question_text,
                    list(map(float, new_emb.tolist()))
                )
                return {"ok": True, "inserted": True, "match_id": None}

            existing_ids = []
            existing_embs = []
            missing_indices = []

            for i, r in enumerate(rows):
                existing_ids.append(r["Id"])
                emb = r.get("Embedding", None)
                if emb is None:
                    missing_indices.append(i)
                    existing_embs.append(None)
                else:
                    existing_embs.append(np.asarray(emb, dtype=float))

            if missing_indices:
                texts_to_embed = [rows[i]["DisplayText"] for i in missing_indices]
                try:
                    new_embs_for_missing = self._embed(texts_to_embed)
                except Exception:
                    new_embs_for_missing = None

                if new_embs_for_missing is not None:
                    for idx, emb_vec in zip(missing_indices, new_embs_for_missing):
                        row_id = rows[idx]["Id"]
                        existing_embs[idx] = emb_vec
                        await conn.execute(
                            'UPDATE "PopularQueries" SET "Embedding" = $1 WHERE "Id" = $2',
                            list(map(float, emb_vec.tolist())),
                            row_id
                        )

            best_sim = -1.0
            best_id = None

            for idx, emb_vec in enumerate(existing_embs):
                if emb_vec is None:
                    continue
                sim = util.cos_sim(new_emb, emb_vec)[0][0].item()
                if sim > best_sim:
                    best_sim = sim
                    best_id = existing_ids[idx]

            if best_sim >= similarity_threshold and best_id is not None:
                await conn.execute(
                    'UPDATE "PopularQueries" SET "Count" = "Count" + 1, "LastAsked" = NOW() WHERE "Id" = $1',
                    best_id
                )
                return {"ok": True, "matched": True, "match_id": best_id, "similarity": float(best_sim)}
            else:
                await conn.execute(
                    """INSERT INTO "PopularQueries" ("DisplayText", "Count", "LastAsked", "Embedding") VALUES ($1, 1, NOW(), $2)""",
                    question_text,
                    list(map(float, new_emb.tolist()))
                )
                return {"ok": True, "matched": False, "match_id": None}

    async def get_top_popular_queries(self, conn, limit: int = 10):
        rows = await conn.fetch(
            """SELECT "DisplayText", "Count", "LastAsked" FROM "PopularQueries" ORDER BY "Count" DESC, "LastAsked" DESC LIMIT $1""",
            limit
        )
        return [
            {
                "text": r["DisplayText"],
                "count": r["Count"],
                "last_asked": r["LastAsked"].isoformat() if r["LastAsked"] else None
            }
            for r in rows
        ]