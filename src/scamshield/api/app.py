"""FastAPI entry point for local SMS screening."""

from __future__ import annotations

import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, Protocol

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scamshield.rules import analyze_message


ROOT = Path(__file__).resolve().parents[3]
SELECTION_PATH = ROOT / "reports" / "sms_model_selection.json"
MODEL_DIR = ROOT / "models" / "sms_distilmbert"
FRONTEND_INDEX = ROOT / "frontend" / "index.html"
FRONTEND_STYLES = ROOT / "frontend" / "styles.css"
FRONTEND_SCRIPT = ROOT / "frontend" / "app.js"
MAX_MESSAGE_LENGTH = 5000
MAX_TOKEN_LENGTH = 256


class Predictor(Protocol):
    model_id: str
    threshold: float

    def predict_spam_probability(self, message: str) -> float: ...


class LocalDistilBertPredictor:
    """Load only the locally saved, hash-verified model selected for this project."""

    def __init__(self, model: torch.nn.Module, tokenizer: object, model_id: str, threshold: float):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_project(cls) -> "LocalDistilBertPredictor":
        selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
        if selection.get("decision_status") != "locked_for_single_test_evaluation":
            raise RuntimeError("The selected SMS model/threshold is not locked.")

        model_info = selection["selected_model"]
        weights_path = ROOT / model_info["weights_file"]
        digest = hashlib.sha256()
        with weights_path.open("rb") as weights:
            for chunk in iter(lambda: weights.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != model_info.get("weights_sha256"):
            raise RuntimeError("The selected model weights do not match the lock record.")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, local_files_only=True
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_id=model_info["model_id"],
            threshold=float(selection["locked_threshold"]),
        )

    def predict_spam_probability(self, message: str) -> float:
        encoded = self.tokenizer(
            message,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**encoded).logits.float()
            return float(torch.softmax(logits, dim=1)[0, 1].item())


class SMSRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: Annotated[str, Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)]

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must contain non-whitespace text")
        return value


class RuleSignal(BaseModel):
    code: str
    title: str
    explanation: str
    details: list[str]


class SMSResponse(BaseModel):
    classification: Literal["spam_like", "not_flagged"]
    spam_probability: float
    threshold: float
    model_id: str
    signal_count: int
    signals: list[RuleSignal]
    disclaimer: str


class HealthResponse(BaseModel):
    status: Literal["ready"]
    model_id: str
    device: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor = getattr(app.state, "predictor_override", None)
    if predictor is None:
        predictor = LocalDistilBertPredictor.from_project()
    app.state.predictor = predictor
    yield


def create_app(predictor: Predictor | None = None) -> FastAPI:
    app = FastAPI(
        title="ScamShield India SMS Screening API",
        description=(
            "Screens a message for generic SMS spam-like patterns and provides transparent "
            "rule indicators. It does not prove fraud or establish India-specific performance."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.predictor_override = predictor

    @app.get("/", include_in_schema=False)
    def frontend() -> HTMLResponse:
        return HTMLResponse(FRONTEND_INDEX.read_text(encoding="utf-8"))

    @app.get("/styles.css", include_in_schema=False)
    def frontend_styles() -> Response:
        return Response(FRONTEND_STYLES.read_text(encoding="utf-8"), media_type="text/css")

    @app.get("/app.js", include_in_schema=False)
    def frontend_script() -> Response:
        return Response(FRONTEND_SCRIPT.read_text(encoding="utf-8"), media_type="application/javascript")

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        service: Predictor | None = getattr(request.app.state, "predictor", None)
        if service is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        device = str(getattr(service, "device", "test-double"))
        return HealthResponse(status="ready", model_id=service.model_id, device=device)

    @app.post("/v1/analyze/sms", response_model=SMSResponse)
    def analyze_sms(payload: SMSRequest, request: Request) -> SMSResponse:
        service: Predictor | None = getattr(request.app.state, "predictor", None)
        if service is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        probability = service.predict_spam_probability(payload.message)
        rule_result = analyze_message(payload.message)
        model_classification: Literal["spam_like", "not_flagged"] = (
            "spam_like" if probability >= service.threshold else "not_flagged"
        )
        return SMSResponse(
            classification=model_classification,
            spam_probability=probability,
            threshold=service.threshold,
            model_id=service.model_id,
            signal_count=int(rule_result["signal_count"]),
            signals=rule_result["signals"],  # type: ignore[arg-type]
            disclaimer=(
                "A spam-like result is a screening signal, not proof of fraud. "
                "A message not flagged is not guaranteed safe. The model is benchmarked on English UCI SMS data; "
                + str(rule_result["disclaimer"])
            ),
        )

    return app


app = create_app()
