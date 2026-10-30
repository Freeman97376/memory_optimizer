import argparse
import os

import uvicorn
from pydantic import SecretStr

from memory_optimizer.config import Settings
from memory_optimizer.storage.postgres import migrate


def main():
    parser = argparse.ArgumentParser(description="Memory Optimizer local service")
    parser.add_argument("action", choices=["migrate", "serve"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--env-file", help="explicit local configuration file")
    parser.add_argument("--flow", choices=["v1", "s3-openai-v1"], default="v1")
    args = parser.parse_args()
    if args.env_file:
        from memory_optimizer.env_file import load_env_file

        load_env_file(args.env_file)
    settings = Settings.from_env()
    if args.action == "migrate":
        migrate(settings.database)
        print("Database migration complete.")
    else:
        settings.validate_service()
        if settings.api_key is None and args.host not in {"127.0.0.1", "::1"}:
            parser.error("anonymous mode can only bind a loopback address")
        from memory_optimizer.api.app import create_app

        services = None
        if args.flow == "s3-openai-v1":
            from memory_optimizer.online import build_online_services
            from memory_optimizer.preflight import PREFLIGHT_SCHEMA

            if settings.schema != PREFLIGHT_SCHEMA:
                parser.error("preflight S3 must reuse its persistent approved-budget schema")
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                parser.error("OPENAI_API_KEY is required")
            services = build_online_services(settings, api_key=SecretStr(key))
        uvicorn.run(
            create_app(settings, services=services),
            host=args.host,
            port=args.port,
            proxy_headers=False,
        )


if __name__ == "__main__":
    main()
