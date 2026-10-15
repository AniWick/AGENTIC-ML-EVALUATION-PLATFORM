import json
from typing import Dict, Any, List, Optional

import chromadb
from app.core.config import settings


class ChromaProjectMemory:
    """Persistent semantic memory for experiments, observations, reports, and questions."""

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or str(settings.chroma_path)

        try:
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(
                name="ml_project_memory",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(
                name="ml_project_memory"
            )

    @staticmethod
    def _clean_metadata(
        project_id: str,
        memory_type: str,
        metadata: Dict[str, Any],
        experiment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        clean_meta = {
            "project_id": str(project_id),
            "type": memory_type,
        }

        if experiment_id is not None:
            clean_meta["experiment_id"] = str(experiment_id)

        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean_meta[key] = value
            else:
                clean_meta[key] = json.dumps(value, default=str)

        return clean_meta

    def store_experiment(
        self,
        project_id: str,
        experiment_id: str,
        summary_text: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Store an experiment summary and its metadata."""

        clean_meta = self._clean_metadata(
            project_id=project_id,
            memory_type="experiment",
            metadata=metadata,
            experiment_id=experiment_id,
        )

        doc_id = f"exp_{project_id}_{experiment_id}"

        self.collection.upsert(
            documents=[summary_text],
            metadatas=[clean_meta],
            ids=[doc_id],
        )

    def store_observation(
        self,
        project_id: str,
        observation_id: str,
        text: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Store an agent observation."""

        clean_meta = self._clean_metadata(
            project_id=project_id,
            memory_type="observation",
            metadata=metadata,
        )

        doc_id = f"obs_{project_id}_{observation_id}"

        self.collection.upsert(
            documents=[text],
            metadatas=[clean_meta],
            ids=[doc_id],
        )

    def query_history(
        self,
        project_id: str,
        query: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant experiment/observation history.

        Only project memory is searched. Chat messages themselves are not
        stored or returned by this method.
        """

        try:
            # Request a few extra results so duplicates can be removed
            # without reducing the useful number of returned memories.
            search_count = max(n_results * 3, n_results)

            results = self.collection.query(
                query_texts=[query],
                where={
                    "$and": [
                        {"project_id": {"$eq": str(project_id)}},
                        {
                            "type": {
                                "$in": [
                                    "experiment",
                                    "observation",
                                ]
                            }
                        },
                    ]
                },
                n_results=search_count,
            )

            if not results:
                return []

            documents = results.get("documents") or [[]]
            metadatas = results.get("metadatas") or [[]]
            distances = results.get("distances") or [[]]

            docs = documents[0] if documents else []
            metas = metadatas[0] if metadatas else []
            dists = distances[0] if distances else []

            retrieved: List[Dict[str, Any]] = []
            seen = set()

            for index, document in enumerate(docs):
                if not document:
                    continue

                metadata = (
                    metas[index]
                    if index < len(metas) and metas[index]
                    else {}
                )

                # Prevent the same generated/stored answer from appearing
                # repeatedly in the context sent to the LLM.
                dedup_key = (
                    str(metadata.get("experiment_id", "")),
                    str(metadata.get("type", "")),
                    document.strip(),
                )

                if dedup_key in seen:
                    continue

                seen.add(dedup_key)

                item = {
                    "content": document,
                    "metadata": metadata,
                }

                if index < len(dists):
                    item["distance"] = dists[index]

                retrieved.append(item)

                if len(retrieved) >= n_results:
                    break

            return retrieved

        except Exception:
            return []


project_memory = ChromaProjectMemory()