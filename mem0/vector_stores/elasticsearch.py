import logging
from typing import Dict, Optional
from urllib.parse import urlparse

from pydantic import BaseModel

try:
    from elasticsearch import Elasticsearch
except ImportError:
    raise ImportError("The 'elasticsearch' library is required. Please install it using 'pip install elasticsearch'.")


from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)

class OutputData(BaseModel):
    id: Optional[str]  # memory id
    score: Optional[float]  # distance
    payload: Optional[Dict]  # metadata


class ESVector(VectorStoreBase):
    def __init__(
            self, 
            collection_name, 
            host="localhost", 
            port=9200,
            username=None,
            password=None,
            embedding_model_dims=1024
        ):
        """Initialize the MilvusDB database.

        Args:
            collection_name (str): Name of the index name (defaults to mem0).
            host (str): es cluster host (defaults to localhost).
            port (int): es cluster port (defaults to 9200).
            username (str): es cluster username (defaults to None).
            password (str): es cluster password (defaults to None).
            embedding_model_dims (int): Dimensions of the embedding model (defaults to 1024).
        """

        self.collection_name = collection_name
        
        parsed_url = urlparse(host)
        if parsed_url.scheme in {"http", "https"}:
            hosts = f"{host}:{port}"
        else:
            hosts = f"http://{host}:{port}"
        if username is None or password is None:
            self.client = Elasticsearch(
                hosts=hosts,
                request_timeout=10000,
                retry_on_timeout=True,
                max_retries=3,
            )
        else:
            self.client = Elasticsearch(
                hosts=hosts,
                basic_auth=(username, password),
                request_timeout=10000,
                retry_on_timeout=True,
                max_retries=3,
            )

        self.embedding_model_dims = embedding_model_dims
        self.create_col(self.collection_name, self.embedding_model_dims, "cosine")



    def create_col(self, name, vector_size, distance):
        if self.client.indices.exists(index=name):
            logger.info(f"Collection {name} already exists. Skipping creation.")
        else:
            body = {
                "mappings": {
                    "_source": {"excludes":["vector"]},
                    "properties": {
                        "id": {"type": "keyword"},
                        "vector": {
                            "type": "dense_vector",
                            "dims": vector_size,
                            "index": True,
                            "similarity": distance
                        },
                        "payload": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "keyword"},
                                "run_id": {"type": "keyword"},
                                "agent_id": {"type": "keyword"},
                            }
                        }
                    }
                }
            }
            self.client.indices.create(index=name, body=body)

    def insert(self, vectors, payloads=None, ids=None):
        for doc_id, vector, payload in zip(ids, vectors, payloads):
            doc = {"id": doc_id, "vector": vector, "payload": payload}
            self.client.index(index=self.collection_name, id=doc_id, body=doc)

    def search(self, query, limit=5, filters=None):
        body = {
                "knn": {
                    "field": "vector",
                    "query_vector": query,
                    "k": limit,
                    "num_candidates": 2 * limit
                    }
                }
        filter_query = self._create_filter(filters)
        if filter_query:
            body["knn"]["filter"] = filter_query
        response = self.client.search(index=self.collection_name, body=body)
        # what if errors
        return self._parse_output(response["hits"]["hits"])

    def delete(self, vector_id):
        self.client.delete(index=self.collection_name, id=vector_id)

    def update(self, vector_id, vector=None, payload=None):
        doc = {}
        if vector:
            doc["vector"] = vector
        if payload:
            doc["payload"] = payload
        if doc:
            self.client.update(index=self.collection_name, id=vector_id, doc=doc)

    def get(self, vector_id):
        response = self.client.get(index=self.collection_name, id=vector_id)
        return OutputData(id=response["_source"]["id"], score=None, payload=response["_source"].get("payload", {}))

    def list_cols(self):
        return self.client.indices.get_alias("*").keys()

    def delete_col(self):
        self.client.indices.delete(index=self.collection_name)

    def col_info(self):
        return self.client.indices.get(index=self.collection_name)

    def list(self, filters=None, limit=None):
        filters = self._create_filter(filters)
        if filters:
            body = {"query": filters}
        else:
            body = {"query": {"match_all": {}}}
        if limit:
            body["size"] = limit
        response = self.client.search(index=self.collection_name, body=body)
        return [self._parse_output(response["hits"]["hits"])]

    def _create_filter(self, filters: Dict[str, str]) -> Dict:
        filter_query = {}
        if filters:
            if len(filters) > 1:
                filter_query["bool"] = {"must": []}
                for key, value in filters.items():
                    filter_query["bool"]["must"].append({"term": {f"payload.{key}": value}})
            else:
                for key, value in filters.items():
                    filter_query = {"term": {f"payload.{key}": value}}
        return filter_query

    def _parse_output(self, hits):
        memories = []
        for hit in hits:
            uid, score, payload = (
                hit["_source"]["id"],
                hit["_score"],
                hit["_source"].get("payload", {}),
            )
            memories.append(OutputData(id=uid, score=score, payload=payload))
        return memories









