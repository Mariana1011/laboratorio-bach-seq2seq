from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import torch
import torch.nn.functional as F
import mlflow.pytorch
from mlflow import MlflowClient
import os

MODEL_NAME = "bach-chorales-mariana"

app = FastAPI(
    title="Bach Chorales API",
    version="1.0",
    root_path=os.getenv("ROOT_PATH", "")
)

client = MlflowClient()

champion = client.get_model_version_by_alias(
    MODEL_NAME,
    "champion"
)

MODEL_VERSION = str(champion.version)

model = mlflow.pytorch.load_model(
    f"models:/{MODEL_NAME}/{MODEL_VERSION}",
    map_location="cpu"
)

model.eval()


class GenerateRequest(BaseModel):
    seed: List[List[int]]
    steps: int = Field(..., ge=1, le=64)
    temperature: float = Field(1.0, ge=0.5, le=1.5)


def validar_seed(seed):
    if len(seed) != 32:
        raise HTTPException(
            status_code=422,
            detail="seed debe tener exactamente 32 pasos"
        )

    for fila in seed:
        if len(fila) != 4:
            raise HTTPException(
                status_code=422,
                detail="Cada paso debe contener exactamente 4 notas"
            )

        for nota in fila:
            if not isinstance(nota, int):
                raise HTTPException(
                    status_code=422,
                    detail="Todas las notas deben ser enteros"
                )

            if nota < 0 or nota > 127:
                raise HTTPException(
                    status_code=422,
                    detail="Las notas deben estar entre 0 y 127"
                )


def generar_bloque(model, contexto, temperature, pasos):
    with torch.no_grad():

        _, h = model.encoder(contexto)

        decoder_input = contexto[:, -1, :]

        generados = []

        for _ in range(pasos):

            logits, h = model.decoder.forward_step(
                decoder_input,
                h
            )

            probs = F.softmax(
                logits / temperature,
                dim=-1
            )

            notas = []

            for voz in range(4):

                nota = torch.multinomial(
                    probs[:, voz, :],
                    num_samples=1
                )

                notas.append(nota)

            siguiente = torch.cat(
                notas,
                dim=1
            )

            generados.append(siguiente)

            decoder_input = siguiente

        return torch.stack(
            generados,
            dim=1
        )


def generar_continuacion(
    model,
    contexto_inicial,
    total_steps,
    temperature
):
    contexto = contexto_inicial.clone()

    generados_totales = []

    faltan = total_steps

    while faltan > 0:

        pasos_bloque = min(
            16,
            faltan
        )

        bloque = generar_bloque(
            model,
            contexto,
            temperature,
            pasos_bloque
        )

        generados_totales.append(
            bloque
        )

        secuencia = torch.cat(
            [contexto, bloque],
            dim=1
        )

        contexto = secuencia[
            :, -32:, :
        ]

        faltan -= pasos_bloque

    return torch.cat(
        generados_totales,
        dim=1
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "version": MODEL_VERSION
    }


@app.post("/generate")
def generate(request: GenerateRequest):

    validar_seed(request.seed)

    contexto = torch.tensor(
        request.seed,
        dtype=torch.long
    ).unsqueeze(0)

    generados = generar_continuacion(
        model,
        contexto,
        total_steps=request.steps,
        temperature=request.temperature
    )

    salida = generados[
        0
    ].cpu().tolist()

    return {
        "generated": salida,
        "model": MODEL_NAME,
        "version": MODEL_VERSION
    }