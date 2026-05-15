# NYS Risk Manager — Engine + Agente para revisión de cuentas prop firm

Sistema deterministico para evaluar cuentas de NYS Markets contra las reglas de los 3 modelos (Instant, 1-Step, 2-Step) y sus fases. Genera veredicto, reporte interno y email al cliente.

## Cómo funciona

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. ENGINE (Python puro, deterministico, sin LLM)                   │
│     rules_engine.py + mt5_parser.py                                 │
│     → Calcula daily DD, total DD, profit target, exposure, payout   │
│     → Output: JSON estructurado con findings + verdict              │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. AGENTE NARRADOR (Anthropic Managed Agents)                      │
│     agent.yaml + run_session.py                                     │
│     → Recibe JSON, NO recalcula                                     │
│     → Output: VEREDICTO + REPORTE INTERNO + EMAIL                   │
└─────────────────────────────────────────────────────────────────────┘
```

El **engine** hace todo el trabajo numérico (auditable, sin variación entre corridas). El **agente** solo escribe los textos legibles para humanos.

## Estructura del proyecto

```
riskmanager/
├── rules_engine.py      Motor: reglas codificadas por modelo+fase
├── mt5_parser.py        Parser de MT5 ReportHistory.html (UTF-16)
├── local_check.py       CLI para análisis local (dry-run sin LLM)
├── run_session.py       CLI que llama al agente Anthropic
├── api.py               HTTP API (FastAPI) para integración con Lovable
├── agent.yaml           Config del agente Anthropic Managed
├── environment.yaml     Sandbox del agente (cloud, networking)
├── seed_memory.py       (legacy) carga reglas a memory store
├── Dockerfile           Para deployar la API
├── requirements.txt     Dependencias Python
├── .env.example         Template de env vars
└── rules/               PDFs originales de NYS por modelo
    ├── instant/rules.pdf
    ├── 1step/rules.pdf
    ├── 1step/live.pdf
    ├── 2step/rules.pdf
    └── 2step/live.pdf
```

## Reglas codificadas (resumen)

| Modelo / Fase | Daily Loss | Max Loss | Profit Target | Min Days | News |
|---|---|---|---|---|---|
| Instant | **3% trailing** intraday | **5% trailing** equity | 3% para payout | — | NO |
| 1-Step Eval | 3% fixed (open) | 6% fixed (initial) | 10% | 3 | SÍ |
| 1-Step Funded | 3% fixed (open) | 6% fixed (initial) | — | — | NO |
| 2-Step Phase 1 | 5% fixed (open) | 10% fixed (initial) | 8% | 3 | SÍ |
| 2-Step Phase 2 | 5% fixed (open) | 10% fixed (initial) | 5% | 3 | SÍ |
| 2-Step Funded | 5% fixed (open) | 10% fixed (initial) | — | — | NO |

**Reglas adicionales chequeadas:**
- **Exposure por símbolo (4%)**: dual-metric — risk-at-stake con SL + margin committed
- **15% Consistency Rule** (instant): mejor día ≤ 15% del profit total
- **Min Profitable Days** (instant): ≥7 días con ≥0.25% del initial en 30d para payout
- **Min Trading Days**: 3 para eval phases, contado por `open_time`
- **Reconciliación**: balance final = initial + sum(net_pnl)
- **Trading day**: reset a 21:00 UTC (server rollover NYS), no calendar day

**Reglas NO chequeadas** (requieren data externa o detección heurística):
- News trading (necesita calendario económico)
- HFT / latency arbitrage / tick scalping
- Copy trading / account sharing / third party
- Patrones de comportamiento (martingala, over-leveraging súbito)

El agente las marca como "merece revisión humana" si detecta sospechas.

## Setup local

```sh
# Requisitos
# - Python 3.11+
# - poppler-utils (solo si vas a usar seed_memory.py con PDFs)
#   - macOS: brew install poppler
#   - Linux: sudo apt install poppler-utils

# Clonar e instalar
git clone <repo-url>
cd riskmanager
pip install -r requirements.txt

# Variables de entorno
cp .env.example .env
# Editar .env con tus valores
```

### Variables de entorno

| Variable | Para qué | Requerido para |
|---|---|---|
| `ANTHROPIC_API_KEY` | API key de Anthropic | Agente + API `/narrate`, `/full` |
| `AGENT_ID` | ID del Managed Agent | Idem |
| `ENV_ID` | ID del Environment | Idem |
| `API_SECRET` | Secret para auth de la API HTTP | API en producción |
| `ALLOWED_ORIGINS` | Dominios permitidos por CORS | API en producción |

## Uso local (CLI)

### Análisis solo (sin LLM, gratis)

```sh
# Instant $10K
python local_check.py trades/ReportHistory9708.html --model instant --size 10000

# 2-step funded $5K
python local_check.py trades/ReportHistory5621.html --model 2step --phase funded

# Con previous payouts (afecta split 80% → 90%)
python local_check.py trades/report.html --model 2step --phase funded --previous-payouts 3

# Output JSON (para feeding a otra app)
python local_check.py trades/report.html --model instant --json > analysis.json
```

### Con el agente narrador

Primero hacer setup del agente (una sola vez):

```sh
# Instalar la CLI de Anthropic
brew install anthropics/tap/ant
# o: go install github.com/anthropics/anthropic-cli/cmd/ant@latest

# Crear agent + environment
export ANTHROPIC_API_KEY=sk-ant-...
export AGENT_ID=$(ant beta:agents create < agent.yaml --transform id -r)
export ENV_ID=$(ant beta:environments create < environment.yaml --transform id -r)

# Guardar en .env
echo "AGENT_ID=$AGENT_ID" >> .env
echo "ENV_ID=$ENV_ID" >> .env
```

Después, en cada revisión:

```sh
source .env
python run_session.py 2step trades/ReportHistory5621.html --phase funded
# → Engine corre primero (local), después el agente narra el resultado
# → Outputs en ./outputs/
```

## API HTTP (para Lovable)

### Correr la API local

```sh
# Desarrollo
uvicorn api:app --reload --port 8000

# Test
curl http://localhost:8000/health
```

### Endpoints

| Endpoint | Auth | Qué hace |
|---|---|---|
| `GET /health` | público | Healthcheck |
| `POST /analyze` | API key | Sube reporte HTML, recibe análisis JSON (sin LLM, rápido y barato) |
| `POST /narrate` | API key | Recibe JSON, devuelve reporte + email (llama al agente) |
| `POST /full` | API key | Combinación: subi reporte y recibí análisis + narración en una sola llamada |

### Ejemplo de llamada

```sh
# Análisis solo
curl -X POST http://localhost:8000/analyze \
  -H "X-API-Key: $API_SECRET" \
  -F "report=@trades/ReportHistory5621.html" \
  -F "model=2step" \
  -F "phase=funded"
```

```json
{
  "model": "2step",
  "phase": "funded",
  "initial_balance": 5000.0,
  "final_balance": 5800.87,
  "verdict": "WARNING",
  "findings": [
    {
      "severity": "warning",
      "rule": "Max Exposure per Symbol (4.0%) — MARGEN",
      "detail": "Pico RISK-AT-STAKE (SL): $182.98 (3.7%) ...",
      "date": "2026-04-20"
    }
  ],
  "payout": {
    "eligible": true,
    "closed_profit": 800.87,
    "profit_split_pct": 80,
    "payout_trader_usd": 640.70,
    "payout_company_usd": 160.17
  },
  "daily_breakdown": [ ... ],
  "metrics": { ... }
}
```

### Deploy de la API

#### Opción A: Railway (más fácil)

```sh
# Tener Railway CLI: brew install railway
railway init
railway up
# Configurar env vars en el dashboard de Railway:
#   ANTHROPIC_API_KEY, AGENT_ID, ENV_ID, API_SECRET, ALLOWED_ORIGINS
```

#### Opción B: Render

1. Crear servicio nuevo en https://render.com → Connect repo
2. Settings:
   - Runtime: Docker
   - Plan: Starter ($7/mo) o Free (con cold starts)
3. Environment: agregar `ANTHROPIC_API_KEY`, `AGENT_ID`, `ENV_ID`, `API_SECRET`, `ALLOWED_ORIGINS`

#### Opción C: Fly.io / Cloud Run / VPS

El Dockerfile que viene en el repo está listo para cualquier plataforma que corra contenedores. Build:

```sh
docker build -t riskmanager-api .
docker run -p 8000:8000 --env-file .env riskmanager-api
```

## Integración con Lovable (frontend React)

### 1. Configurar el endpoint en tu proyecto Lovable

En el chat de Lovable:

```
Quiero integrar una API externa de risk management.
La API está en: https://riskmanager-api.tudominio.com
Autenticación: header X-API-Key con secret almacenado en variables de entorno (no hardcodear).
Crear un servicio TypeScript que llame a POST /analyze con FormData (file + campos).
```

Lovable va a generar algo como:

```typescript
// src/lib/riskmanager.ts
const API_URL = import.meta.env.VITE_RISKMANAGER_API_URL;
const API_KEY = import.meta.env.VITE_RISKMANAGER_API_KEY;

export interface AnalysisResult {
  model: string;
  phase: string | null;
  initial_balance: number;
  final_balance: number;
  verdict: "PASS" | "BREACH" | "WARNING";
  findings: Finding[];
  payout: PayoutInfo;
  daily_breakdown: DailyStats[];
  metrics: Record<string, number>;
}

export interface Finding {
  severity: "breach" | "warning" | "info";
  rule: string;
  detail: string;
  date?: string;
}

export interface PayoutInfo {
  eligible: boolean;
  closed_profit?: number;
  profit_split_pct?: number;
  payout_trader_usd?: number;
  payout_company_usd?: number;
  blockers?: string[];
}

export async function analyzeReport(params: {
  file: File;
  model: "instant" | "1step" | "2step";
  phase?: "evaluation" | "phase1" | "phase2" | "funded";
  size?: number;
  previousPayouts?: number;
}): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("report", params.file);
  formData.append("model", params.model);
  if (params.phase) formData.append("phase", params.phase);
  if (params.size) formData.append("size", String(params.size));
  if (params.previousPayouts) formData.append("previous_payouts", String(params.previousPayouts));

  const res = await fetch(`${API_URL}/analyze`, {
    method: "POST",
    headers: { "X-API-Key": API_KEY },
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Analysis failed");
  }

  return res.json();
}

export async function narrateAnalysis(analysis: AnalysisResult): Promise<string> {
  const res = await fetch(`${API_URL}/narrate`, {
    method: "POST",
    headers: {
      "X-API-Key": API_KEY,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ analysis }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Narration failed");
  }

  const data = await res.json();
  return data.narration;
}
```

### 2. UI sugerida (Lovable prompt)

```
Crear una página /review con:
- Form para subir archivo .html (MT5 ReportHistory)
- Select: model (instant | 1step | 2step)
- Select condicional: phase (depende del model)
- Input number: account size USD
- Botón "Analizar" → llama analyzeReport()
- Mostrar resultado:
  - Card grande con VERDICT (color: verde PASS, amarillo WARNING, rojo BREACH)
  - Tabla con balance inicial/final, P&L, profit %
  - Lista de findings agrupados por severity
  - Card de payout con monto al trader si elegible
  - Botón "Generar reporte + email" → llama narrateAnalysis()
  - Mostrar narración en bloque markdown render
```

### 3. Variables de entorno en Lovable

En el panel de Lovable / Supabase:

```
VITE_RISKMANAGER_API_URL=https://tu-api.railway.app
VITE_RISKMANAGER_API_KEY=<el mismo API_SECRET que pusiste en la API>
```

⚠️ **Importante sobre seguridad**: poner el API key en `VITE_*` lo expone al cliente. Para producción, mejor:
- Hacer un Supabase Edge Function que reciba la request del frontend (autenticada con JWT del user), agregue el `X-API-Key` desde un secret, y llame a tu API Python. Así el secret nunca llega al browser.

Prompt para Lovable:

```
Crear un Supabase Edge Function llamado "analyze-account" que:
1. Reciba multipart/form-data del frontend (file + model + phase + size)
2. Verifique JWT del usuario (Supabase Auth)
3. Reenvíe la request a NEXT_PUBLIC_RISKMANAGER_API_URL/analyze
   agregando el header X-API-Key desde el secret RISKMANAGER_API_KEY
4. Devuelva la respuesta tal cual al frontend
```

## Costos estimados

| Operación | Costo aproximado |
|---|---|
| `POST /analyze` (engine solo) | $0 (local Python) — solo costo del hosting |
| `POST /narrate` (con agente) | ~$0.10-0.50 por revisión (depende del tamaño del JSON) |
| `POST /full` | Igual que narrate |

El engine corre en milisegundos. El agente toma 5-20 segundos.

## Audit y troubleshooting

### Verificar que el motor calcule bien

```sh
# Corré el local_check sobre un reporte y compará con el summary de MT5
python local_check.py trades/<reporte>.html --model <model> --phase <phase>

# La sección "Reconciliación con MT5" muestra si el P&L cuadra.
# Si NO cuadra (✗), revisar:
# 1. El reporte está completo (todas las posiciones)
# 2. El modelo/phase correctos
# 3. El size es el balance inicial real (o detectado del Deal type='balance')
```

### Logs de la API

```sh
# Local
uvicorn api:app --reload --log-level debug

# Railway / Render
# Ver el dashboard del servicio → logs
```

### Validar setup del agente

```sh
# Listar agents existentes
ant beta:agents list --transform '{id,name,model,version}' --format jsonl

# Probar una sesión simple
ant beta:sessions create --agent $AGENT_ID --environment-id $ENV_ID
```

## Roadmap / Cosas para iterar

- [ ] Soporte para CSV/XLSX exports de otros brokers (actualmente solo HTML MT5)
- [ ] News calendar integration para verificar trades en ventana ±5 min
- [ ] HFT / latency detection (análisis estadístico de inter-trade timing)
- [ ] Multi-cuenta: detectar coordinación entre accounts del mismo trader
- [ ] Equity tick data (en lugar de solo balance al cierre) si MT5 lo expone
- [ ] Dashboard de payouts históricos y compliance trends

## Licencia

Privado — uso interno de NYS Markets.
