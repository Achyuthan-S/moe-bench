

# MoE-Bench 

**Una evaluación abierta para el colapso de expertos y la eficiencia de enrutamiento en LLMs de Mezcla de Expertos Esparcida (MoE).**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Mac MPS Validated](https://img.shields.io/badge/Mac%20MPS-Validated-brightgreen.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> MoE-Bench es un **kit de herramientas instalable mediante pip** para medir la entropía de enrutamiento, la utilización de expertos y el colapso en LLMs MoE esparcidos; no es un ranking de precisión en tareas posteriores. Está listo para ejecutarse en un Mac de 24 GB.

---

## ¿Por qué MoE-Bench?

Los modelos MoE esparcidos enrutan tokens a un subconjunto de expertos por capa. Cuando el enrutamiento colapsa —un pequeño número de expertos monopoliza todos los tokens—, el modelo desperdicia capacidad y se degrada en silencio. Las evaluaciones existentes miden *qué* produce un modelo, no *cómo* se comporta su enrutamiento.

MoE-Bench llena ese vacío:

- **Entropía de enrutamiento** (Shannon H) por capa y por dominio
- **Distribución de utilización de expertos** y detección de colapso (τ configurable)
- **Intervalos de confianza bootstrap** en las estadísticas de enrutamiento
- **Entrenamiento L_div** — un regularizador de piso de entropía que evita el colapso sin dañar la precisión
- **Tres arquitecturas validadas** listas para usar: OLMoE, JetMoE, Qwen1.5-MoE

---

## Resultados Rápidos

Los tres modelos evaluados con las **mismas indicaciones matemáticas locales** (40 muestras, 256 tokens).

| Modelo | Colapso matemático (τ=0.40) | Entropía media | Utilización pico del mejor experto |
|---|---|---|---|
| OLMoE-1B-7B | **50 %** de las capas | 3.66 | ~82 % |
| JetMoE-8B | 0 % | 1.84 | ~19 % |
| Qwen1.5-MoE-A2.7B | 0 % | 3.30 | ~16 % |

**Hallazgo clave:** el colapso depende de la arquitectura, no es una propiedad universal de los modelos MoE. OLMoE colapsa significativamente en matemáticas; JetMoE y Qwen no.

Tabla completa con intervalos de confianza bootstrap de capa al 95 % → [`results/ROUTING_PROFILE_TABLE.md`](results/ROUTING_PROFILE_TABLE.md)

---

## L_div: ¿Arreglar el colapso afecta la precisión?

Probamos el ajuste fino LoRA con L_div en OLMoE contra un subconjunto de MMLU de 12 materias (5-shot, 25 ejemplos/materia):

| Modelo | Precisión macro | IC al 95 % |
|---|---|---|
| Base OLMoE | 51.7 % | [44.4 %, 59.0 %] |
| + L_div LoRA | 52.0 % | [46.4 %, 57.6 %] |
| Δ | +0.33 pp | Los ICs se superponen → no significativo |

**L_div no afecta la precisión en tareas posteriores** — el piso de entropía regulariza el enrutamiento sin degradar la calidad de la salida. Se necesita una ejecución completa de MMLU con 57 tareas antes de hacer cualquier afirmación de precisión; esto es una comprobación de consistencia direccional.

---

## Cómo difiere L_div de las pérdidas de enrutamiento existentes

| Pérdida | Limitación | L_div |
|---|---|---|
| Switch auxiliary loss | El balance de carga ≠ alta entropía | Piso de entropía en capas colapsadas |
| Router z-loss | Estabiliza logits, no diversidad | `max(0, τ − H_l)` por capa |
| Load-balancing loss | Pocos expertos aún pueden dominar | Shannon H por capa |

---

## Instalación

```bash
git clone https://github.com/Achyuthan-S/moe-bench.git
cd moe-bench
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[train,eval,demo,dev]"
```

---

## Inicio Rápido

```python
import moe_bench
from moe_bench.device import load_causal_lm, model_input_device

model, tokenizer, _ = load_causal_lm("allenai/OLMoE-1B-7B-0924", device="mps")
stats = moe_bench.attach(model)

inputs = tokenizer("Hello", return_tensors="pt").to(model_input_device(model))
model(**inputs)

print(stats.to_dataframe()[["layer", "mean_entropy", "collapsed"]])
moe_bench.detach(model)
```

---

## Interfaz de Línea de Comandos (CLI)

```bash
# Analyze routing on a math domain
moe-bench analyze \
  --model allenai/OLMoE-1B-7B-0924 \
  --dataset math \
  --device mps \
  --max-samples 40 \
  --max-length 256 \
  --output results/routing_stats_olmoe_math.json

# Cross-model domain profiles
python scripts/run_domain_profiles.py --domains math general code

# Regenerate leaderboard table + bootstrap CIs
python scripts/enrich_routing_reports.py
python scripts/update_leaderboard.py

# OLMoE only: L_div training + MMLU eval
python scripts/run_mmlu_ldiv_comparison.py --ldiv-only

# Gradio demo
python scripts/demo_gradio.py
```

---

## Estructura del Repositorio

```
moe_bench/          # Core library: attach, hooks, metrics, routing_uncertainty, train
scripts/            # run_domain_profiles, run_jetmoe_mac, update_leaderboard, ...
results/            # routing_stats_*.json, ROUTING_PROFILE_TABLE.md, ANALYSIS_NOTES.md
CONTRIBUTING.md     # Hook guide for adding new MoE architectures
```

---

## Archivos de Salida Clave

| Ruta | Contenido |
|---|---|
| `results/ROUTING_PROFILE_TABLE.md` | Todos los perfiles con intervalos de confianza bootstrap al 95 % |
| `results/ANALYSIS_NOTES.md` | Análisis en profundidad de OLMoE |
| `results/CROSS_MODEL_NOTES.md` | Comparación matemática OLMoE vs JetMoE |
| `results/routing_stats_jetmoe_math.json` | Datos de enrutamiento matemático de JetMoE |
| `results/routing_stats_qwen15_moe_math.json` | Datos de enrutamiento matemático de Qwen1.5-MoE (CPU) |
| `results/lm_eval/mmlu_ldiv_comparison.json` | MMLU base vs L_div |
| `results/ablation_comparison.csv` | Ablaciones de enrutamiento (LoRA sintético) |

---

## Hardware y Ámbito

| Plataforma | Modelos admitidos |
|---|---|
| Mac MPS 24GB | OLMoE, JetMoE (secuencial); Qwen suele ir en CPU después de JetMoE |
| Cloud GPU (opcional) | Mixtral-8x7B mediante `moe-bench analyze --4bit` — no validado en esta versión |

**Notas sobre el ámbito:**
- El entrenamiento LoRA con L_div, las ablation y MMLU son **exclusivos de OLMoE** en esta versión.
- JetMoE y Qwen son líneas base **solo de perfil de enrutamiento** (sin ajuste fino L_div).
- Compare siempre los modelos en **filas coincidentes por dominio** (`math` / `code` / `general`), no en filas sintéticas de prueba.
- Las ejecuciones MoE grandes en la nube (p. ej., Mixtral-8x7B a escala de paper) están fuera de ámbito aquí. La CLI las admite mediante `--4bit` cuando hay acceso a GPU.

---

## Contribuciones

El soporte de hooks para nuevas arquitecturas está en [`CONTRIBUTING.md`](CONTRIBUTING.md). Agregar un modelo se resume en identificar el nombre del módulo de compuerta/enrutador y registrar un hook; el resto del kit es independiente de la arquitectura.

Las solicitudes de pull (PR) son bienvenidas.

---

## Citación

```bibtex
@misc{sivasankar2026moebench,
  title        = {MoE-Bench: Open Benchmark for Expert Collapse and Routing Efficiency in Sparse MoE LLMs},
  author       = {Sivasankar, Achyuthan and {MoE-Bench Contributors}},
  year         = {2026},
  howpublished = {Software},
  url          = {https://github.com/Achyuthan-S/moe-bench},
  note         = {Routing benchmark toolkit; MMLU and cloud-scale Mixtral results reported as directional or out-of-scope}
}
```

---

## Licencia

Apache 2.0 — ver [`LICENSE`](LICENSE).
