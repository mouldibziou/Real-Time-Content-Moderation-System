import json
from unittest import TestCase, IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch
from configparser import ConfigParser
from app.db.urls import database_url


class DatabaseUrlTests(TestCase):
    def test_provider_url_preserves_encoded_credentials_and_tls(self):
        value = "postgres://user:p%40ss%25word@managed.example/toxicity?sslmode=require"
        asynchronous = database_url(value, asynchronous=True)
        synchronous = database_url(value, asynchronous=False)
        self.assertEqual(asynchronous.drivername, "postgresql+asyncpg")
        self.assertEqual(asynchronous.password, "p@ss%word")
        self.assertEqual(asynchronous.query, {"ssl": "require"})
        self.assertEqual(synchronous.query, {"sslmode": "require"})
        config = ConfigParser()
        config.add_section("alembic")
        rendered = synchronous.render_as_string(hide_password=False)
        config.set("alembic", "sqlalchemy.url", rendered.replace("%", "%%"))
        self.assertEqual(config.get("alembic", "sqlalchemy.url"), rendered)

    def test_async_url_can_be_used_by_migrations(self):
        url = database_url("postgresql+asyncpg://u:p@host/db?ssl=verify-full", asynchronous=False)
        self.assertEqual(url.drivername, "postgresql+psycopg2")
        self.assertEqual(url.query, {"sslmode": "verify-full"})

    def test_rejects_other_databases_and_conflicting_tls_options(self):
        for value in ("sqlite:///db", "postgresql://host/db?ssl=require&sslmode=require"):
            with self.assertRaises(ValueError):
                database_url(value, asynchronous=True)


class ReadinessTests(IsolatedAsyncioTestCase):
    async def test_raw_connection_errors_use_the_database_503_handler(self):
        from app.db.engine import get_db
        from sqlalchemy.exc import SQLAlchemyError
        for error in (ConnectionRefusedError("unavailable"), TimeoutError("timed out")):
            with self.subTest(error=type(error).__name__):
                factory = MagicMock()
                factory.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
                factory.return_value.__aexit__ = AsyncMock(return_value=False)
                with patch("app.db.engine.async_session", factory):
                    dependency = get_db()
                    await anext(dependency)
                    with self.assertRaises(SQLAlchemyError):
                        await dependency.athrow(error)

    async def test_database_outage_returns_503(self):
        from app.api.routes import get_readiness
        session = AsyncMock()
        session.execute.side_effect = ConnectionError("database unavailable")
        factory = MagicMock()
        factory.return_value.__aenter__ = AsyncMock(return_value=session)
        factory.return_value.__aexit__ = AsyncMock(return_value=None)
        with patch("app.api.routes.async_session", factory):
            response = await get_readiness(MagicMock(is_ready=lambda: True))
        self.assertEqual(response.status_code, 503)
        self.assertEqual(json.loads(response.body), {"is_ready": False})

    async def test_unready_model_does_not_check_database(self):
        from app.api.routes import get_readiness
        with patch("app.api.routes.async_session") as factory:
            response = await get_readiness(MagicMock(is_ready=lambda: False))
        self.assertEqual(response.status_code, 503)
        factory.assert_not_called()
