from sqlalchemy.engine import URL, make_url


def database_url(value: str, *, asynchronous: bool) -> URL:
    """Use a provider's PostgreSQL URL with the appropriate installed driver."""
    if value.startswith("postgres://"):
        value = "postgresql://" + value[len("postgres://"):]
    url = make_url(value)
    if url.get_backend_name() != "postgresql":
        raise ValueError("DATABASE_URL must refer to PostgreSQL")
    query = dict(url.query)
    source, target = ("sslmode", "ssl") if asynchronous else ("ssl", "sslmode")
    if source in query:
        if target in query:
            raise ValueError("Specify only one of ssl and sslmode in DATABASE_URL")
        query[target] = query.pop(source)
    return url.set(drivername="postgresql+asyncpg" if asynchronous else "postgresql+psycopg2", query=query)
