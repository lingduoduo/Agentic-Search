from pydantic import BaseModel


def shallow_model_dump(model: BaseModel) -> dict:
    """Return model fields as a dict without deep serialization."""
    return {k: v for k, v in model.__dict__.items() if not k.startswith("_")}
