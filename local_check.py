"""Verifica un reporte MT5 contra las reglas NYS — TODO determinístico, sin LLM.

Corre el rules_engine.py contra el HTML parseado y muestra:
  - Métricas calculadas (balance, P&L, DD, profit target, etc.)
  - Cada regla evaluada (PASS / WARNING / BREACH) con evidencia
  - Veredicto final
  - Reconciliación contra el summary del propio MT5

Uso:
    pip install beautifulsoup4 lxml
    python local_check.py <reporte.html> --model <m> [--phase <p>]
                          [--size <usd>] [--server-utc-offset <horas>]
                          [--json]

Ejemplos:
    python local_check.py trades/ReportHistory9708.html --model instant --size 10000
    python local_check.py trades/cuenta.html --model 2step --phase phase1
    python local_check.py trades/x.html --model 1step --phase funded --json > out.json
"""

import argparse
import json
import sys
from pathlib import Path

from mt5_parser import parse, parse_number, read_html
from rules_engine import RULES, evaluate, to_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("report", type=Path)
    p.add_argument("--model", required=True, choices=["instant", "1step", "2step"])
    p.add_argument(
        "--phase",
        choices=["evaluation", "phase1", "phase2", "funded"],
        help="Para 1-step: evaluation|funded. Para 2-step: phase1|phase2|funded. Instant: omitir.",
    )
    p.add_argument("--size", type=int, help="Tamaño cuenta en USD; si se omite, se detecta del reporte")
    p.add_argument("--server-utc-offset", type=int, default=0,
                   help="Offset horario del server MT5 vs UTC (default 0)")
    p.add_argument("--previous-payouts", type=int, default=0,
                   help="Cantidad de payouts previos aprobados (afecta split: 80%% → 90%% tras 3)")
    p.add_argument("--json", action="store_true", help="Output JSON puro (para feeding al agente)")
    return p.parse_args()


def normalize_phase(model: str, phase: str | None) -> str | None:
    if model == "instant":
        return None
    if model == "1step":
        if phase in (None, "evaluation"):
            return "evaluation"
        if phase == "funded":
            return "funded"
        sys.exit(f"Fase inválida para 1-step: {phase} (válidas: evaluation, funded)")
    if model == "2step":
        if phase in ("phase1", "phase2", "funded"):
            return phase
        sys.exit(f"Fase requerida para 2-step: phase1 | phase2 | funded")
    return phase


def print_human(html: str, result, summary: dict[str, str]) -> None:
    """Imprime resultado en formato legible para humanos."""
    print(f"\n╔══════════════════════════════════════════════════════════════════════════════╗")
    print(f"║  RISK MANAGER — {result.model.upper():<8} {(result.phase or '').upper():<10}                                    ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════════╝\n")

    print(f"Balance inicial:    ${result.initial_balance:>12,.2f}")
    print(f"Balance final:      ${result.final_balance:>12,.2f}  "
          f"(P&L: ${result.metrics['total_pnl']:+,.2f}  "
          f"{result.metrics['total_pnl_pct']:+.2f}%)")
    print(f"Trades:             {result.n_trades:>13}")
    print(f"Trading days:       {result.metrics['n_trading_days']:>13}")
    if result.period_start:
        print(f"Período:            {result.period_start:%Y-%m-%d} → {result.period_end:%Y-%m-%d}")

    print("\nDía         #tr   open$        P&L$    close$    daily%    peak$    DD-peak%   trailing$")
    print("─" * 100)
    for d in result.daily_breakdown:
        flag = ""
        if abs(d.daily_pct) >= result.rules_applied.get("daily_loss_pct", 999) and d.daily_pct < 0:
            flag += " ⚠DAILY"
        if result.rules_applied.get("drawdown_type") == "trailing" and d.close_balance < d.trailing_dd_limit:
            flag += " ⚠TRAIL"
        dd_peak_pct = (1 - d.close_balance / d.running_peak) * 100 if d.running_peak else 0
        trailing_str = f"{d.trailing_dd_limit:>9,.2f}" if d.trailing_dd_limit > 0 else "—"
        print(
            f"{d.trading_day}  {d.n_trades:>3}  "
            f"{d.open_balance:>9,.2f}  {d.day_pnl:>+9,.2f}  "
            f"{d.close_balance:>9,.2f}  {d.daily_pct:>+6.2f}%  "
            f"{d.running_peak:>9,.2f}  {dd_peak_pct:>6.2f}%   {trailing_str}{flag}"
        )

    # Findings
    print("\n" + "─" * 100)
    print("REGLAS EVALUADAS\n")
    by_severity = {"breach": [], "warning": [], "info": []}
    for f in result.findings:
        by_severity[f.severity].append(f)

    if by_severity["breach"]:
        print("✗ BREACHES")
        for f in by_severity["breach"]:
            print(f"  → {f.rule}")
            print(f"     {f.detail}\n")
    if by_severity["warning"]:
        print("⚠ WARNINGS")
        for f in by_severity["warning"]:
            print(f"  → {f.rule}")
            print(f"     {f.detail}\n")
    if by_severity["info"]:
        print("✓ INFO")
        for f in by_severity["info"]:
            print(f"  → {f.rule}: {f.detail}")

    # Veredicto
    print("\n" + "═" * 100)
    color = {"PASS": "✓", "WARNING": "⚠", "BREACH": "✗"}
    print(f"\n  {color[result.verdict]}  VEREDICTO: {result.verdict}\n")
    print("═" * 100)

    # Reconciliación con MT5
    if "Total Net Profit" in summary:
        reported = parse_number(summary["Total Net Profit"])
        calc = result.metrics["total_pnl"]
        diff = abs(reported - calc)
        match = "✓" if diff < 0.5 else "✗"
        print(f"\nReconciliación con MT5: reporta ${reported:+,.2f} / calculado ${calc:+,.2f}  {match}")

    if "Balance Drawdown Maximal" in summary:
        print(f"MT5 reporta Balance DD: {summary['Balance Drawdown Maximal']}")
        print(f"Calculado DD desde peak: {result.metrics['max_drawdown_from_peak_pct']:.2f}%")
        print("  (diferencia esperada: MT5 mide equity intraday; aquí solo balance al cierre)")

    # Payout
    p = result.payout
    if p:
        print("\n" + "─" * 100)
        print("PAYOUT")
        print("─" * 100)
        if p.get("eligible"):
            print(f"  ✓ ELEGIBLE")
            print(f"  Profit cerrado:        ${p['closed_profit']:>+10,.2f}  ({p['closed_profit_pct']:+.2f}%)")
            print(f"  Profit split:           {p['profit_split_pct']}%")
            print(f"  Trader recibe:         ${p['payout_trader_usd']:>+10,.2f}")
            print(f"  Empresa retiene:       ${p['payout_company_usd']:>+10,.2f}")
            if p.get("min_payout_usd"):
                print(f"  Min payout:            ${p['min_payout_usd']}")
            print(f"  Días desde 1er trade:   {p['days_since_first_trade']}  (mínimo {p.get('min_days_required','?')})")
        else:
            print(f"  ✗ NO ELEGIBLE")
            if "reason" in p:
                print(f"  Razón: {p['reason']}")
            for blocker in p.get("blockers", []):
                print(f"    - {blocker}")
            if "closed_profit" in p:
                print(f"\n  Profit cerrado: ${p['closed_profit']:+,.2f}")
                print(f"  Días desde 1er trade: {p['days_since_first_trade']}")


def main() -> None:
    args = parse_args()
    if not args.report.is_file():
        sys.exit(f"No existe: {args.report}")

    phase = normalize_phase(args.model, args.phase)
    if (args.model, phase) not in RULES and args.model != "instant":
        sys.exit(f"Combinación no soportada: model={args.model} phase={phase}")

    html = read_html(args.report)
    initial_balance, trades, summary = parse(html)

    if args.size:
        if abs(args.size - initial_balance) > 1:
            print(
                f"⚠ Override: --size ${args.size:,} sobrescribe balance detectado "
                f"${initial_balance:,.2f}",
                file=sys.stderr,
            )
        initial_balance = float(args.size)

    result = evaluate(
        model=args.model,
        phase=phase,
        initial_balance=initial_balance,
        trades=trades,
        server_utc_offset_hours=args.server_utc_offset,
        n_previous_payouts=args.previous_payouts,
    )

    if args.json:
        # JSON-only output: para feeding al agente
        out = result.to_dict()
        out["mt5_summary"] = summary
        print(json.dumps(out, indent=2, default=str))
    else:
        print_human(html, result, summary)


if __name__ == "__main__":
    main()
