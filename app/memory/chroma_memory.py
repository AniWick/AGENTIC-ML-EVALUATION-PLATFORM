import json
from typing import Dict, Any, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings

class ChromaProjectMemory:
    """Persistent semantic memory for experiments, observations, reports, and questions."""

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or str(settings.chroma_path)
        try:
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(
                name="ml_project_memory",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            # Fallback in-memory client if persistence fails
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(name="ml_project_memory")

    def store_experiment(
        self,
        project_id: str,
        experiment_id: str,
        summary_text: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Stores an experiment summary with its metadata for semantic recall."""
        # Ensure metadata values are str/int/float/bool
        clean_meta = {
            "project_id": str(project_id),
            "experiment_id": str(experiment_id),
            "type": "experiment"
        }
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = json.dumps(v)

        doc_id = f"exp_{project_id}_{experiment_id}"
        self.collection.upsert(
            documents=[summary_text],
            metadatas=[clean_meta],
            ids=[doc_id]
        )

    def store_observation(
        self,
        project_id: str,
        observation_id: str,
        text: str,
        metadata: Dict[str, Any]
    ) -> None:
        clean_meta = {
            "project_id": str(project_id),
            "type": "observation"
        }
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = json.dumps(v)

        doc_id = f"obs_{project_id}_{observation_id}"
        self.collection.upsert(
            documents=[text],
            metadatas=[clean_meta],
            ids=[doc_id]
        )

    def query_history(self, project_id: str, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Queries semantic memory for experiments or past observations matching the query."""
        try:
            results = self.collection.query(
                query_texts=[query],
                where={"project_id": project_id},
                n_results=n_results
            )
            retrieved = []
            if results and "documents" in results and len(results["documents"]) > 0:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if "metadatas" in results else []
                for d, m in zip(docs, metas):
                    retrieved.append({"content": d, "metadata": m})
            return retrieved
        except Exception:
            return []

project_memory = ChromaProjectMemory()

