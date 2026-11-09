"""Deployment helpers and environment configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DeploymentSettings:
    environment: str
    database_url: str | None
    app_base_url: str | None
    secret_key: str | None

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


def get_deployment_settings() -> DeploymentSettings:
    return DeploymentSettings(
        environment=os.getenv("SMARTTINNITUS_ENV", "development"),
        database_url=os.getenv("DATABASE_URL"),
        app_base_url=os.getenv("APP_BASE_URL"),
        secret_key=os.getenv("SMARTTINNITUS_SECRET_KEY"),
    )
