import os
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from app.settings import load_runtime_settings


class WorkerConfigurationTests(IsolatedAsyncioTestCase):
    async def test_worker_uses_deployment_overrides(self):
        from app.worker import main

        class StartupIntercepted(Exception):
            pass

        with patch.dict(os.environ, {
            "TOXICITY_MODEL_DIR": "/release/torch",
            "TOXICITY_ONNX_MODEL_DIR": "/release/onnx",
            "TOXICITY_INFERENCE_BACKEND": "torch",
        }), patch("app.worker_redis.create_worker_redis", return_value=AsyncMock()), \
             patch("app.runtime.ApplicationRuntime", side_effect=StartupIntercepted) as runtime:
            with self.assertRaises(StartupIntercepted):
                await main()
            self.assertEqual(runtime.call_args.kwargs["settings"], load_runtime_settings())
