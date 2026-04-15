from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class PipelineRunRequest(BaseModel):
    target_date: date = Field(default_factory=date.today)
    refresh_aggregates: bool = True
    rebuild_features: bool = True


# Dashboard update-job action keys (see ``UPDATE_JOB_ACTION_LABELS`` in ``update_job_sequences``).
UpdateAction = str

UpdateJobStatus = Literal["queued", "running", "succeeded", "failed"]
