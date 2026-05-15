"""Motor determinístico de reglas del prop firm NYS.

Calcula todas las métricas y evaluaciones contra reglas SIN LLM. Output es un
dict JSON-serializable que el agente recibe para escribir reporte + email.

Lo que el motor SÍ verifica (numérico, determinístico):
  - Daily loss (con trading day 21:00 UTC reset)
  - Max loss fijo (1-step / 2-step / live)
  - Trailing loss (instant)
  - Profit target (eval phases)
  - Min trading days (eval)
  - 15% consistency rule (instant)
  - Min 7 profitable days en 30d (instant, para payout)
  - Max risk per trade idea (heurística: agrupa por symbol+side dentro de
    ventana de 5 min; sin stop-loss real, usa el adverse del open-close)
  - Max exposure per instrument (suma de volúmenes * precio por símbolo)
  - Account activity (>=1 trade por 30d)

Lo que el motor NO verifica (requiere juicio humano o data externa):
  - HFT, latency arbitrage, server spamming
  - News trading (necesita calendario de eventos)
  - Copy trading / third party / account sharing
  - Patrones de comportamiento (martingala, over-leveraging súbito)

Esos quedan flageados como "manual review needed" para que el equipo decida.

Importante sobre Equity vs Balance:
Las reglas son EQUITY-based (incluyen floating P&L). Con MT5 ReportHistory
solo tenemos trades cerrados. El motor calcula sobre balance-close y agrega
WARNING explícito de que el daily/total DD real intraday pudo ser peor.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Literal, Optional


# ────────────────────────────────────────────────────────────────────────────
# Configuración por modelo y fase
# ────────────────────────────────────────────────────────────────────────────

Model = Literal["instant", "1step", "2step"]
Phase = Literal["evaluation", "phase1", "phase2", "funded"]


RULES = {
    # Instant: trailing daily + trailing overall, sin evaluation
    ("instant", None): {
        "daily_loss_pct": 3.0,
        "daily_loss_type": "trailing",         # límite se mueve con peak intraday
        "drawdown_type": "trailing",
        "drawdown_pct": 5.0,
        "profit_target_pct": None,             # no hay target; 3% es solo para payout
        "min_profit_for_payout_pct": 3.0,
        "min_trading_days": None,
        "min_profitable_days_30d": 7,
        "profitable_day_threshold_pct": 0.25,
        "consistency_max_day_pct_of_total": 15.0,
        "max_risk_per_idea_pct": 2.0,
        "max_exposure_per_symbol_pct": 4.0,
        "max_floating_loss_pct": 1.0,
        "trade_idea_window_minutes": 5,
        "max_leverage": 50,
        "news_trading_allowed": False,
    },
    # 1-step evaluation: fixed daily + fixed total, target 10%
    ("1step", "evaluation"): {
        "daily_loss_pct": 3.0,
        "daily_loss_type": "fixed",            # snapshot a las 21:00 UTC
        "drawdown_type": "fixed",
        "drawdown_pct": 6.0,
        "profit_target_pct": 10.0,
        "min_trading_days": 3,
        "max_leverage": 30,
        "news_trading_allowed": True,
    },
    # 2-step phase 1: fixed daily + fixed total, target 8%
    ("2step", "phase1"): {
        "daily_loss_pct": 5.0,
        "daily_loss_type": "fixed",
        "drawdown_type": "fixed",
        "drawdown_pct": 10.0,
        "profit_target_pct": 8.0,
        "min_trading_days": 3,
        "max_leverage": 100,
        "news_trading_allowed": True,
    },
    # 2-step phase 2: fixed daily + fixed total, target 5%
    ("2step", "phase2"): {
        "daily_loss_pct": 5.0,
        "daily_loss_type": "fixed",
        "drawdown_type": "fixed",
        "drawdown_pct": 10.0,
        "profit_target_pct": 5.0,
        "min_trading_days": 3,
        "max_leverage": 100,
        "news_trading_allowed": True,
    },
    # 1-step funded (live): daily 3%, max loss 6% fixed. News NOT allowed.
    ("1step", "funded"): {
        "daily_loss_pct": 3.0,
        "daily_loss_type": "fixed",
        "drawdown_type": "fixed",
        "drawdown_pct": 6.0,
        "profit_target_pct": None,
        "max_risk_per_idea_pct": 3.0,
        "max_exposure_per_symbol_pct": 4.0,
        "trade_idea_window_minutes": 5,
        "news_trading_allowed": False,
    },
    # 2-step funded (live): daily 5%, max loss 10% fixed. News NOT allowed.
    ("2step", "funded"): {
        "daily_loss_pct": 5.0,
        "daily_loss_type": "fixed",
        "drawdown_type": "fixed",
        "drawdown_pct": 10.0,
        "profit_target_pct": None,
        "max_risk_per_idea_pct": 3.0,
        "max_exposure_per_symbol_pct": 4.0,
        "trade_idea_window_minutes": 5,
        "news_trading_allowed": False,
    },
}


# ────────────────────────────────────────────────────────────────────────────
# Modelos de datos
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class Trade:
    open_time: datetime   # server time (assumed UTC unless told otherwise)
    close_time: datetime
    symbol: str
    side: str             # buy | sell
    volume: float
    open_price: float
    close_price: float
    commission: float
    swap: float
    profit: float

    @property
    def net_pnl(self) -> float:
        return self.profit + self.commission + self.swap


@dataclass
class DailyStats:
    trading_day: date          # day-of-reset (21:00 UTC start)
    n_trades: int
    open_balance: float
    day_pnl: float
    close_balance: float
    daily_pct: float           # day_pnl / open_balance * 100
    running_peak: float
    daily_dd_limit: float      # 3% / 5% absoluto sobre open_balance
    trailing_dd_limit: float   # solo si trailing


@dataclass
class Finding:
    """Una violación o advertencia."""
    severity: Literal["breach", "warning", "info"]
    rule: str
    detail: str
    date: Optional[date] = None
    actual: Optional[float] = None
    limit: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.date is not None:
            d["date"] = self.date.isoformat()
        return d


@dataclass
class EngineResult:
    model: Model
    phase: Optional[Phase]
    initial_balance: float
    final_balance: float
    period_start: Optional[datetime]
    period_end: Optional[datetime]
    n_trades: int
    rules_applied: dict
    daily_breakdown: list[DailyStats]
    findings: list[Finding]
    metrics: dict
    verdict: Literal["PASS", "BREACH", "WARNING"]

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "phase": self.phase,
            "initial_balance": self.initial_balance,
            "final_balance": self.final_balance,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "n_trades": self.n_trades,
            "rules_applied": self.rules_applied,
            "daily_breakdown": [
                {
                    **asdict(d),
                    "trading_day": d.trading_day.isoformat(),
                }
                for d in self.daily_breakdown
            ],
            "findings": [f.to_dict() for f in self.findings],
            "metrics": self.metrics,
            "verdict": self.verdict,
        }


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def trading_day(ts: datetime, server_utc_offset_hours: int = 0) -> date:
    """Convierte un timestamp en server time al trading day (reset 21:00 UTC).

    Un trading day va desde 21:00 UTC del día N hasta 21:00 UTC del día N+1.
    Convención: el trading day se etiqueta con la fecha del día N+1 (el día
    que dura más horas) — esto coincide con cómo MT5 muestra el día siguiente
    al hacer el rollover.
    """
    utc = ts - timedelta(hours=server_utc_offset_hours)
    if utc.hour >= 21:
        return (utc + timedelta(days=1)).date()
    return utc.date()


# ────────────────────────────────────────────────────────────────────────────
# Cálculos por regla
# ────────────────────────────────────────────────────────────────────────────


def compute_daily_breakdown(
    initial_balance: float,
    trades: list[Trade],
    rules: dict,
    server_utc_offset_hours: int,
) -> tuple[list[DailyStats], dict[date, list[Trade]]]:
    """Agrupa trades por trading day, computa balance corriente + DD limits.

    Devuelve (lista de DailyStats, dict trades_por_día) — el segundo se
    reusa después para chequear daily loss con detalle intraday.
    """
    by_day: dict[date, list[Trade]] = defaultdict(list)
    for t in trades:
        by_day[trading_day(t.close_time, server_utc_offset_hours)].append(t)
    # Trades del día ordenados por close_time (necesario para intraday tracking)
    for day in by_day:
        by_day[day].sort(key=lambda t: t.close_time)

    daily_loss_pct = rules.get("daily_loss_pct", 0)
    is_total_trailing = rules.get("drawdown_type") == "trailing"
    drawdown_pct = rules.get("drawdown_pct", 0)

    breakdown: list[DailyStats] = []
    running = initial_balance
    peak = initial_balance
    for day in sorted(by_day):
        day_trades = by_day[day]
        open_bal = running
        day_pnl = sum(t.net_pnl for t in day_trades)
        running = open_bal + day_pnl
        peak = max(peak, running)
        breakdown.append(
            DailyStats(
                trading_day=day,
                n_trades=len(day_trades),
                open_balance=open_bal,
                day_pnl=day_pnl,
                close_balance=running,
                daily_pct=(day_pnl / open_bal * 100) if open_bal else 0.0,
                running_peak=peak,
                daily_dd_limit=open_bal * daily_loss_pct / 100,
                trailing_dd_limit=peak * (1 - drawdown_pct / 100) if is_total_trailing else 0.0,
            )
        )
    return breakdown, by_day


def check_daily_loss(
    daily: list[DailyStats],
    trades_by_day: dict[date, list[Trade]],
    rules: dict,
) -> list[Finding]:
    """Verifica daily loss. Dos modos según rules['daily_loss_type']:

    - "fixed":    límite = open_balance × (1 - X%) constante todo el día
    - "trailing": límite = max(intraday_running_balance) × (1 - X%) — el
                  punto de referencia sube con cada nuevo peak intraday
    """
    findings: list[Finding] = []
    limit_pct = rules.get("daily_loss_pct")
    if limit_pct is None:
        return findings
    is_trailing = rules.get("daily_loss_type") == "trailing"
    factor = 1 - limit_pct / 100

    for d in daily:
        if d.n_trades == 0:
            continue
        # Reconstruir trayectoria intraday del balance
        running = d.open_balance
        peak_intraday = d.open_balance
        worst_dd = 0.0          # % desde peak intraday
        breach_at: Trade | None = None
        breach_floor = 0.0

        breach_balance = 0.0
        breach_ref = 0.0
        for t in trades_by_day.get(d.trading_day, []):
            running += t.net_pnl
            if is_trailing:
                peak_intraday = max(peak_intraday, running)
            ref = peak_intraday if is_trailing else d.open_balance
            floor = ref * factor
            dd_pct = (ref - running) / ref * 100 if ref else 0
            if dd_pct > worst_dd:
                worst_dd = dd_pct
            if running < floor and breach_at is None:
                breach_at = t
                breach_floor = floor
                breach_balance = running
                breach_ref = ref

        if breach_at is not None:
            kind = "trailing" if is_trailing else "fixed"
            ref_text = (
                f"trailing peak intraday ${breach_ref:,.2f}"
                if is_trailing else
                f"open balance ${d.open_balance:,.2f}"
            )
            findings.append(Finding(
                severity="breach",
                rule=f"Max Daily Loss {limit_pct}% ({kind})",
                detail=(
                    f"El {d.trading_day} tras el trade cerrado a las "
                    f"{breach_at.close_time:%H:%M:%S} el balance cayó a "
                    f"${breach_balance:,.2f}, debajo del floor ${breach_floor:,.2f} "
                    f"({limit_pct}% sobre {ref_text}). DD intraday peak {worst_dd:.2f}%. "
                    "Cálculo sobre balance al cierre — equity intraday con "
                    "floating P&L pudo haber tocado el floor incluso antes."
                ),
                date=d.trading_day,
                actual=worst_dd,
                limit=limit_pct,
            ))
        elif worst_dd >= limit_pct - 1.0:
            kind = "trailing" if is_trailing else "fixed"
            findings.append(Finding(
                severity="warning",
                rule=f"Max Daily Loss {limit_pct}% ({kind}) — NEAR LIMIT",
                detail=(
                    f"El {d.trading_day}: peor DD intraday {worst_dd:.2f}% "
                    f"({kind}), a menos de 1pp del límite {limit_pct}%. "
                    "Con equity intraday (floating P&L) el límite pudo haberse cruzado."
                ),
                date=d.trading_day,
                actual=worst_dd,
                limit=limit_pct,
            ))
    return findings


def check_max_loss(
    daily: list[DailyStats],
    initial_balance: float,
    rules: dict,
) -> list[Finding]:
    findings: list[Finding] = []
    dd_type = rules.get("drawdown_type")
    dd_pct = rules.get("drawdown_pct", 0)
    if not dd_type or not dd_pct:
        return findings

    if dd_type == "fixed":
        limit_balance = initial_balance * (1 - dd_pct / 100)
        for d in daily:
            if d.close_balance <= limit_balance:
                findings.append(Finding(
                    severity="breach",
                    rule=f"Max Loss {dd_pct}% (fixed)",
                    detail=(
                        f"El día {d.trading_day} el balance cerró en "
                        f"${d.close_balance:,.2f}, por debajo del piso fijo "
                        f"${limit_balance:,.2f} ({dd_pct}% del initial)."
                    ),
                    date=d.trading_day,
                    actual=d.close_balance,
                    limit=limit_balance,
                ))
                break
    elif dd_type == "trailing":
        for d in daily:
            if d.close_balance < d.trailing_dd_limit:
                findings.append(Finding(
                    severity="breach",
                    rule=f"Max Trailing Loss {dd_pct}%",
                    detail=(
                        f"El día {d.trading_day} el balance cerró en "
                        f"${d.close_balance:,.2f}, por debajo del trailing "
                        f"${d.trailing_dd_limit:,.2f} "
                        f"(peak ${d.running_peak:,.2f} × {1-dd_pct/100:.2f})."
                    ),
                    date=d.trading_day,
                    actual=d.close_balance,
                    limit=d.trailing_dd_limit,
                ))
                break
        # Buffer warning: si quedaste muy cerca del trailing al final
        if daily and not any(f.rule.startswith("Max Trailing") for f in findings):
            last = daily[-1]
            buffer_pct = (last.close_balance - last.trailing_dd_limit) / last.running_peak * 100
            if buffer_pct < 1.0:
                findings.append(Finding(
                    severity="warning",
                    rule=f"Max Trailing Loss {dd_pct}% (NEAR LIMIT)",
                    detail=(
                        f"Buffer al trailing actual: {buffer_pct:.2f}% del peak. "
                        f"Equity ${last.close_balance:,.2f} vs trailing "
                        f"${last.trailing_dd_limit:,.2f}."
                    ),
                ))
    return findings


def check_profit_target(
    daily: list[DailyStats],
    initial_balance: float,
    rules: dict,
) -> list[Finding]:
    target_pct = rules.get("profit_target_pct")
    if target_pct is None or not daily:
        return []
    final = daily[-1].close_balance
    target_balance = initial_balance * (1 + target_pct / 100)
    profit = final - initial_balance
    profit_pct = profit / initial_balance * 100
    if final >= target_balance:
        return [Finding(
            severity="info",
            rule=f"Profit Target {target_pct}%",
            detail=(
                f"Profit target alcanzado: balance final ${final:,.2f} "
                f"(profit {profit_pct:.2f}%, objetivo {target_pct}%)."
            ),
            actual=profit_pct,
            limit=target_pct,
        )]
    return [Finding(
        severity="warning",
        rule=f"Profit Target {target_pct}%",
        detail=(
            f"Profit target NO alcanzado: profit {profit_pct:.2f}% "
            f"(${profit:+,.2f}), objetivo {target_pct}% (${initial_balance*target_pct/100:,.2f})."
        ),
        actual=profit_pct,
        limit=target_pct,
    )]


def check_min_trading_days(
    daily: list[DailyStats],
    rules: dict,
) -> list[Finding]:
    min_days = rules.get("min_trading_days")
    if min_days is None:
        return []
    if len(daily) >= min_days:
        return [Finding(
            severity="info",
            rule=f"Min Trading Days ({min_days})",
            detail=f"{len(daily)} trading days con actividad (cumple mínimo {min_days}).",
            actual=len(daily),
            limit=min_days,
        )]
    return [Finding(
        severity="warning",
        rule=f"Min Trading Days ({min_days})",
        detail=f"Solo {len(daily)} trading days con actividad — requiere {min_days}.",
        actual=len(daily),
        limit=min_days,
    )]


def check_consistency_15(
    daily: list[DailyStats],
    rules: dict,
) -> list[Finding]:
    max_pct = rules.get("consistency_max_day_pct_of_total")
    if max_pct is None or not daily:
        return []
    total_profit = sum(d.day_pnl for d in daily if d.day_pnl > 0)
    if total_profit <= 0:
        return []
    best = max(daily, key=lambda d: d.day_pnl)
    if best.day_pnl <= 0:
        return []
    pct_of_total = best.day_pnl / total_profit * 100
    if pct_of_total > max_pct:
        return [Finding(
            severity="warning",
            rule=f"{max_pct}% Consistency Rule",
            detail=(
                f"El mejor día ({best.trading_day}) generó ${best.day_pnl:,.2f}, "
                f"que es {pct_of_total:.1f}% del profit total positivo "
                f"(${total_profit:,.2f}). Excede el {max_pct}%. "
                "Bloquea payout hasta que el ratio baje (más días positivos)."
            ),
            date=best.trading_day,
            actual=pct_of_total,
            limit=max_pct,
        )]
    return [Finding(
        severity="info",
        rule=f"{max_pct}% Consistency Rule",
        detail=f"Mejor día = {pct_of_total:.1f}% del profit total positivo (límite {max_pct}%).",
        actual=pct_of_total,
        limit=max_pct,
    )]


def check_min_profitable_days(
    daily: list[DailyStats],
    initial_balance: float,
    rules: dict,
) -> list[Finding]:
    required = rules.get("min_profitable_days_30d")
    threshold_pct = rules.get("profitable_day_threshold_pct", 0)
    if required is None or not daily:
        return []
    threshold_amount = initial_balance * threshold_pct / 100
    # 30-day rolling window backwards from last trade
    last = daily[-1].trading_day
    cutoff = last - timedelta(days=30)
    in_window = [d for d in daily if d.trading_day > cutoff]
    profitable = [d for d in in_window if d.day_pnl >= threshold_amount]
    if len(profitable) >= required:
        return [Finding(
            severity="info",
            rule=f"Min Profitable Days ({required} en 30d)",
            detail=(
                f"{len(profitable)} días rentables (≥${threshold_amount:,.2f}) "
                f"en últimos 30 días — cumple para payout."
            ),
            actual=len(profitable),
            limit=required,
        )]
    return [Finding(
        severity="warning",
        rule=f"Min Profitable Days ({required} en 30d)",
        detail=(
            f"Solo {len(profitable)} días rentables (≥${threshold_amount:,.2f} = "
            f"{threshold_pct}% del initial) en últimos 30 días. "
            f"Requiere {required} para payout."
        ),
        actual=len(profitable),
        limit=required,
    )]


def check_exposure_per_symbol(
    trades: list[Trade],
    initial_balance: float,
    rules: dict,
) -> list[Finding]:
    """Exposición = suma de volume×close_price por símbolo en cualquier momento simultáneo.

    Aproximación simple: para cada apertura, calcular cuánta exposición había
    abierta por símbolo en ese momento (trades que aún no se cerraron).
    """
    max_pct = rules.get("max_exposure_per_symbol_pct")
    if max_pct is None:
        return []
    limit_usd = initial_balance * max_pct / 100

    # Para cada trade, calcular cuánto del mismo símbolo estaba abierto en su open_time
    findings: list[Finding] = []
    by_symbol: dict[str, float] = defaultdict(float)
    worst_per_symbol: dict[str, tuple[datetime, float]] = {}

    # Eventos: open (+exposure), close (-exposure), ordenados temporalmente
    events: list[tuple[datetime, str, float, str]] = []  # (time, action, exposure_usd, symbol)
    for t in trades:
        exp = t.volume * t.open_price
        events.append((t.open_time, "open", exp, t.symbol))
        events.append((t.close_time, "close", -exp, t.symbol))
    events.sort(key=lambda e: e[0])

    open_exp: dict[str, float] = defaultdict(float)
    for ts, action, delta, symbol in events:
        open_exp[symbol] += delta
        if action == "open":
            if open_exp[symbol] > worst_per_symbol.get(symbol, (None, 0))[1]:
                worst_per_symbol[symbol] = (ts, open_exp[symbol])

    for symbol, (ts, peak_exp) in worst_per_symbol.items():
        if peak_exp > limit_usd:
            findings.append(Finding(
                severity="warning",
                rule=f"Max Exposure per Symbol ({max_pct}%)",
                detail=(
                    f"Pico de exposición en {symbol}: ${peak_exp:,.2f} el "
                    f"{ts:%Y-%m-%d %H:%M} — excede ${limit_usd:,.2f} ({max_pct}% "
                    f"del initial ${initial_balance:,.2f}). NOTA: cálculo en USD "
                    "notional con precio de apertura — no incluye apalancamiento."
                ),
                date=ts.date(),
                actual=peak_exp,
                limit=limit_usd,
            ))
    return findings


def check_trade_ideas(
    trades: list[Trade],
    initial_balance: float,
    rules: dict,
) -> list[Finding]:
    """Agrupa trades en 'trade ideas' (mismo símbolo + dirección dentro de ventana).

    Sin tick data ni stop-loss real per-trade, aproximamos el "riesgo" del idea
    como la mayor pérdida bruta que sufrió antes de cerrar.
    """
    max_pct = rules.get("max_risk_per_idea_pct")
    window_min = rules.get("trade_idea_window_minutes", 5)
    if max_pct is None:
        return []
    limit_usd = initial_balance * max_pct / 100

    # Ordenar por open_time y agrupar por (symbol, side) si están dentro de la ventana
    findings: list[Finding] = []
    grouped: list[list[Trade]] = []
    for t in sorted(trades, key=lambda x: x.open_time):
        placed = False
        for group in grouped:
            last = group[-1]
            if (
                last.symbol == t.symbol
                and last.side == t.side
                and (t.open_time - last.open_time).total_seconds() < window_min * 60
            ):
                group.append(t)
                placed = True
                break
        if not placed:
            grouped.append([t])

    for group in grouped:
        # Aproximación de "riesgo": net P&L del grupo si terminó negativo, o el peor
        # P&L individual del grupo si alguno fue muy negativo.
        worst_loss = min((t.net_pnl for t in group), default=0)
        total_pnl = sum(t.net_pnl for t in group)
        # Usar la métrica más conservadora (mayor magnitud de pérdida observada)
        observed_loss = -min(0, total_pnl, worst_loss)
        if observed_loss > limit_usd:
            t0 = group[0]
            findings.append(Finding(
                severity="warning",
                rule=f"Max Risk per Trade Idea ({max_pct}%)",
                detail=(
                    f"Trade idea en {t0.symbol} {t0.side} iniciada {t0.open_time:%Y-%m-%d %H:%M} "
                    f"({len(group)} entradas en {window_min}min). Pérdida observada "
                    f"${observed_loss:,.2f} — excede límite ${limit_usd:,.2f} ({max_pct}%). "
                    "APROXIMACIÓN: sin stop-loss en data, usamos peor P&L observado."
                ),
                date=t0.open_time.date(),
                actual=observed_loss,
                limit=limit_usd,
            ))
    return findings


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────


def evaluate(
    model: Model,
    phase: Optional[Phase],
    initial_balance: float,
    trades: list[Trade],
    server_utc_offset_hours: int = 0,
) -> EngineResult:
    """Corre todas las reglas aplicables y devuelve resultado estructurado."""
    key = (model, phase if model != "instant" else None)
    if key not in RULES:
        raise ValueError(
            f"Combinación no soportada: model={model}, phase={phase}. "
            f"Válidas: {list(RULES.keys())}"
        )
    rules = RULES[key]
    daily, by_day = compute_daily_breakdown(initial_balance, trades, rules, server_utc_offset_hours)

    findings: list[Finding] = []
    findings += check_daily_loss(daily, by_day, rules)
    findings += check_max_loss(daily, initial_balance, rules)
    findings += check_profit_target(daily, initial_balance, rules)
    findings += check_min_trading_days(daily, rules)
    findings += check_consistency_15(daily, rules)
    findings += check_min_profitable_days(daily, initial_balance, rules)
    findings += check_exposure_per_symbol(trades, initial_balance, rules)
    findings += check_trade_ideas(trades, initial_balance, rules)

    # Verdict
    has_breach = any(f.severity == "breach" for f in findings)
    has_warning = any(f.severity == "warning" for f in findings)
    verdict = "BREACH" if has_breach else ("WARNING" if has_warning else "PASS")

    final_balance = daily[-1].close_balance if daily else initial_balance
    period_start = trades[0].open_time if trades else None
    period_end = trades[-1].close_time if trades else None

    metrics = {
        "total_pnl": final_balance - initial_balance,
        "total_pnl_pct": (final_balance - initial_balance) / initial_balance * 100,
        "n_trading_days": len(daily),
        "best_day_pnl": max((d.day_pnl for d in daily), default=0),
        "worst_day_pnl": min((d.day_pnl for d in daily), default=0),
        "best_day_pct": max((d.daily_pct for d in daily), default=0),
        "worst_day_pct": min((d.daily_pct for d in daily), default=0),
        "peak_balance": max((d.running_peak for d in daily), default=initial_balance),
        "max_drawdown_from_peak_pct": (
            max((1 - d.close_balance / d.running_peak for d in daily), default=0) * 100
        ),
    }

    return EngineResult(
        model=model,
        phase=phase,
        initial_balance=initial_balance,
        final_balance=final_balance,
        period_start=period_start,
        period_end=period_end,
        n_trades=len(trades),
        rules_applied=rules,
        daily_breakdown=daily,
        findings=findings,
        metrics=metrics,
        verdict=verdict,
    )


def to_json(result: EngineResult, indent: int | None = 2) -> str:
    return json.dumps(result.to_dict(), indent=indent, default=str)
