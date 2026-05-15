# NYS Risk Manager API

Backend Python (FastAPI) que recibe trades + métricas ya parseados desde el frontend Lovable y devuelve el audit estructurado generado por Claude (Anthropic Messages API + tool calling).

## Arquitectura

```
┌────────────────────────────── Frontend (Lovable / TanStack Start) ──────────────────────────────┐
│                                                                                                 │
│  /reviewer                                                                                      │
│   ├─ User sube ReportHistory.html                                                               │
│   ├─ parseReport()  → trades[] (en el browser)                                                  │
│   ├─ computeMetrics() → DeterministicMetrics  (TODO en TS, autoritativo)                        │
│   ├─ detectFraudSignals() → FraudSignal[]                                                       │
│   └─ analyzeWithRiskManager()  ── HTTPS ──┐                                                     │
│                                            │  POST /audit                                       │
│                                            │  Header: X-API-Key                                 │
│                                            │  Body: { account, preset, trades, metrics }        │
└────────────────────────────────────────────┼────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌──────────────────────────────────── API Python (Railway) ───────────────────────────────────────┐
│                                                                                                 │
│  api.py                                                                                         │
│   ├─ Valida X-API-Key contra API_SECRET                                                         │
│   ├─ Pasa el contexto a Claude Messages API                                                     │
│   ├─ Fuerza tool_choice=submit_audit_result (structured output garantizado)                     │
│   └─ Devuelve { finalDecision, severity, executiveSummary, breaches,                            │
│                 warnings, evidenceTable, internalRecommendation,                                │
│                 emailSubject, emailBody }                                                       │
└────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │ api.anthropic.com    │
                                  │ claude-opus-4-7      │
                                  └──────────────────────┘
```

**Importante**: la API NO recalcula ningún número. El frontend ya hace todo el cálculo determinístico (trades, P&L, drawdown, exposure, fraud signals, etc.) y se lo pasa a Claude para que **explique, contextualice y genere el email** — no para que recalcule.

## Setup en Railway

### 1. Variables de entorno

En tu proyecto Railway → tab **Variables**:

| Variable | Cómo obtenerla | Ejemplo |
|---|---|---|
| `ANTHROPIC_API_KEY` | console.anthropic.com → API Keys → Create | `sk-ant-api03-AbCd...` |
| `API_SECRET` | `openssl rand -hex 32` (en tu terminal) | `a3f7c9e2b8d4...` (32 bytes hex) |
| `ALLOWED_ORIGINS` | Tu URL de Lovable production + dev | `https://tu-app.lovable.app,http://localhost:5173` |

Opcionales (defaults razonables):

| Variable | Default | Para qué |
|---|---|---|
| `ANTHROPIC_MODEL` | `claude-opus-4-7` | Cambiar a `claude-sonnet-4-6` ahorra ~50% si querés |
| `ANTHROPIC_MAX_TOKENS` | `8000` | Suficiente para audit + email + tablas |
| `MAX_TRADES_IN_PROMPT` | `300` | Cap del sample que se manda al modelo |
| `PORT` | `8000` (Railway lo setea) | No tocar |

### 2. Verificar el deploy

Una vez seteadas las vars, Railway redeploya solo. Verificá:

```bash
# Healthcheck (no auth) — debe devolver configured=true para anthropic_key y api_secret
curl https://riskmanager-production.up.railway.app/health
```

Esperado:
```json
{
  "status": "ok",
  "service": "risk-manager-api",
  "version": "2.0.0",
  "configured": {
    "anthropic_key": true,
    "api_secret": true,
    "model": "claude-opus-4-7"
  }
}
```

Si `anthropic_key: false` o `api_secret: false`, falta cargar esa variable.

### 3. Test real

```bash
export API="https://riskmanager-production.up.railway.app"
export KEY="<tu-API_SECRET>"

curl -X POST $API/audit \
  -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "account": {
      "accountNumber": "TEST-001",
      "traderName": "Test Trader",
      "accountType": "TWO_STEP",
      "phase": "LIVE",
      "accountSize": 5000,
      "initialBalance": 5000,
      "serverTimezone": "UTC",
      "dailyResetTime": "21:00"
    },
    "preset": {
      "id": "two-step-live",
      "name": "2-Step Live",
      "description": "Standard 2-step live rules",
      "dailyLossPercent": 5,
      "maxLossPercent": 10,
      "profitTargetPercent": 0,
      "maxExposurePerSymbolPercent": 4,
      "consistencyRulePercent": 0,
      "prohibitedStrategies": []
    },
    "trades": [],
    "metrics": {
      "totalClosedPnL": 800.87,
      "tradingDaysCount": 13,
      "profitTargetReached": "NOT_ENOUGH_DATA",
      "maxClosedLossPercent": 4.92,
      "tradesWithoutSL": []
    }
  }'
```

Devuelve JSON con `data.finalDecision`, `data.emailBody`, `data.internalRecommendation`, etc. — el shape exacto que `riskmanager.functions.ts` espera.

## Setup en Lovable

### 1. Variables de entorno en Lovable

En tu proyecto Lovable → Settings → Environment Variables:

| Variable | Valor |
|---|---|
| `RISKMANAGER_API_URL` | `https://riskmanager-production.up.railway.app` |
| `RISKMANAGER_API_KEY` | El **mismo `API_SECRET`** que pusiste en Railway |

> ⚠️ **Importante**: `RISKMANAGER_API_KEY` debe estar como **server-side env var** (no `VITE_*`), porque tu `riskmanager.functions.ts` la usa en un `createServerFn` (server function). Así nunca llega al browser.

### 2. Verificá que el frontend la lee

En tu repo Lovable hay `src/lib/riskmanager.functions.ts`. Tiene esto:

```typescript
const baseUrl = process.env.RISKMANAGER_API_URL;
const envKey = process.env.RISKMANAGER_API_KEY;
```

Si esas dos están seteadas en Lovable, ya está conectado. Probá desde la UI:

1. Abrí `/reviewer` en tu app
2. Subí un `ReportHistory.html`
3. Llená los campos del form
4. Click **Analyze**
5. La sección de análisis debe poblarse con `finalDecision`, breaches, email body, etc.

## Endpoints

Los 4 endpoints **hacen lo mismo** — son aliases para compatibilidad con el `riskmanager.functions.ts` del frontend que acepta `endpoint = "analyze" | "full" | "narrate" | "audit"`.

| Endpoint | Auth | Descripción |
|---|---|---|
| `GET  /health` | público | Status + config (chequear si `anthropic_key` y `api_secret` están true) |
| `POST /audit` | `X-API-Key` | Audit canonical |
| `POST /analyze` | `X-API-Key` | Alias de `/audit` |
| `POST /full` | `X-API-Key` | Alias de `/audit` (lo que el frontend usa por default) |
| `POST /narrate` | `X-API-Key` | Alias de `/audit` |

### Request schema

```typescript
{
  account: {
    accountNumber: string,
    traderName: string,
    traderEmail?: string,
    accountType: "ONE_STEP" | "TWO_STEP" | "INSTANT",
    phase: "PHASE_1" | "PHASE_2" | "LIVE",
    accountSize: number,
    initialBalance: number,
    currentBalance?: number,
    currentEquity?: number,
    requestedPayout?: number,
    serverTimezone: string,
    dailyResetTime: string
  },
  preset: {
    id: string,
    name: string,
    description: string,
    profitTargetPercent?: number,
    dailyLossPercent?: number,
    maxLossPercent?: number,
    maxRiskPerTradeIdeaPercent?: number,
    maxExposurePerSymbolPercent?: number,
    consistencyRulePercent?: number,
    minTradingDays?: number,
    prohibitedStrategies: string[]
  },
  trades: Trade[],          // los trades parseados por el frontend
  metrics: DeterministicMetrics  // ya calculados por el frontend
}
```

### Response schema

```typescript
{
  ok: true,
  data: {
    finalDecision: "PASSED" | "WARNING" | "BREACH" | "MANUAL_REVIEW",
    severity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
    executiveSummary: string,
    confirmedBreaches: RuleAnalysis[],
    estimatedBreaches: RuleAnalysis[],
    warnings: RuleAnalysis[],
    notEnoughData: RuleAnalysis[],
    ruleByRuleAnalysis: RuleAnalysis[],
    evidenceTable: EvidenceRow[],
    internalRecommendation: "APPROVE PAYOUT" | "REJECT PAYOUT" |
                            "PARTIAL APPROVAL" | "MANUAL REVIEW REQUIRED",
    emailSubject: string,
    emailBody: string
  },
  latencyMs: number,
  model: "claude-opus-4-7",
  usage: { input_tokens, output_tokens, ... },
  trades_dropped: number   // si trades > MAX_TRADES_IN_PROMPT, cuántos se muestrearon
}
```

### Error response

```typescript
{
  ok: false,
  error: string,
  status: number  // 401 / 403 / 422 / 502 / 504
}
```

## Costos

Cada audit usa ~50K-150K tokens input (depende de cuántos trades) + ~3-5K output:

| Modelo | Input/1M | Output/1M | Costo típico por audit |
|---|---|---|---|
| **claude-opus-4-7** (default) | $5 | $25 | **$0.30 – $0.85** |
| claude-sonnet-4-6 | $3 | $15 | $0.15 – $0.45 |
| claude-haiku-4-5 | $1 | $5 | $0.05 – $0.15 |

Para 100 audits/mes con Opus: ~$30-85/mes. Cambiá a Sonnet en `ANTHROPIC_MODEL` para ahorrar ~50%.

## Desarrollo local

```bash
# Clonar y entrar
git clone <repo>
cd riskmanager

# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Editar .env con tus valores

# Correr
uvicorn api:app --reload --port 8000

# Probar
curl http://localhost:8000/health
```

## Troubleshooting

### `anthropic_key: false` en /health
La env var `ANTHROPIC_API_KEY` no está cargada. En Railway → Variables → agregarla → Railway redeploya solo.

### `Invalid API key` (HTTP 403) cuando llamás desde Lovable
El `RISKMANAGER_API_KEY` que pusiste en Lovable no coincide con el `API_SECRET` que pusiste en Railway. Copiá el mismo valor en ambos lados.

### `Anthropic API error (401)` (HTTP 502)
Tu `ANTHROPIC_API_KEY` no es válida. Verificá en console.anthropic.com que la key existe y tiene billing activado.

### `Anthropic API error (429)` (HTTP 502)
Rate limit hit. Para uso masivo, considerá Batches API o tier paid más alto en console.anthropic.com.

### `Anthropic API error (500/529)` (HTTP 502)
Server-side error de Anthropic. Reintentar (idempotente). Si es persistente, ver status.anthropic.com.

### CORS error desde el browser
Agregá tu URL de Lovable a `ALLOWED_ORIGINS` en Railway. Si es solo desde server functions (lo recomendado), CORS no aplica.

### `Model did not call the tool` (HTTP 502)
Raro pero posible si el contexto está malformado. Verificar que `account`, `preset`, `trades`, `metrics` se mandan con las shapes correctas (ver schema arriba). Logs en Railway tienen detalles.

### Latencia alta (>30s)
Normal para audits grandes (muchos trades). Si pasa 45s consistentemente, bajar `MAX_TRADES_IN_PROMPT` a 200 o usar Sonnet en lugar de Opus.

## Estructura del repo

```
riskmanager/
├── api.py              ← Toda la API (FastAPI + Anthropic client + tool schema)
├── requirements.txt    ← anthropic, fastapi, uvicorn, pydantic
├── Dockerfile          ← Para Railway / Render / Fly.io / Docker local
├── .env.example        ← Template de env vars
└── README.md           ← Este archivo
```

Repo minimalista — todo el cálculo determinístico vive en el frontend Lovable (`src/lib/payout-*.ts`). Esta API es solo el adapter HTTP que llama a Claude.

## Roadmap

- [ ] Soporte de streaming (SSE) para mostrar el reporte mientras Claude genera
- [ ] Cache de audits con mismo hash de input (ahorra costos si reanalyzan)
- [ ] Webhooks para notificar BREACH automáticamente (Slack/Discord)
- [ ] Multi-tenant (per-user `API_SECRET`)
- [ ] Soporte explícito de Batches API para reviews masivas

## Licencia

Privado — uso interno.
