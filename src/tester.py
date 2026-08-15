import db
from embeddings import embed_text


def test_embedding() -> None:
    vector = embed_text("hello world")
    print(f"embedding length: {len(vector)}")
    print(f"first 5 values: {vector[:5]}")


def test_db_init() -> None:
    client = db.init_client(url=db.QDRANT_URL)
    db.ensure_collection(client, db.COLLECTION_NAME)
    print(f"collection '{db.COLLECTION_NAME}' ready at {db.QDRANT_URL}")


if __name__ == "__main__":
    test_embedding()
    test_db_init()
