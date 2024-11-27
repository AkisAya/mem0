from typing import Any, Dict

from pydantic import BaseModel, Field, model_validator

class ElasticSearchConfig(BaseModel):
    collection_name: str = Field("mem0", description="Name of the index")
    host: str = Field("localhost", description="host")
    port: int = Field(9200, description="port")
    username: str = Field(None, description="username")
    password: str = Field(None, description="password")
    embedding_model_dims: int = Field(1024, description="Dimension of the embedding vector")


    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        allowed_fields = set(cls.model_fields.keys())
        input_fields = set(values.keys())
        extra_fields = input_fields - allowed_fields
        if extra_fields:
            raise ValueError(
                f"Extra fields not allowed: {', '.join(extra_fields)}. Please input only the following fields: {', '.join(allowed_fields)}"
            )
        return values

    model_config = {
        "arbitrary_types_allowed": True,
    }
