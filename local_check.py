"""Verifica localmente un reporte MT5 antes de gastar una sesión del agente.

Parsea el HTML de MetaTrader 5 ReportHistory, extrae los trades cerrados,
detecta el balance inicial, y calcula las métricas críticas:
  - P&L día a día
  - Daily drawdown (% sobre balance al inicio del día)
  - Drawdown total acumulado
  - Reconciliación del balance final reportado

NO emite veredicto — solo muestra las cifras que el agente va a usar. Si
estos números coinciden con lo que esperás, el agente debería emitir el
mismo veredicto que harías vos manualmente.

Uso:
    pip install beautifulsoup4
    python local_check.py <archivo.html>

Ejemplo:
    python local_check.py ./trades/ReportHistory9708.html
"""

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


@dataclass
class Trade:
    close_time: datetime
    symbol: str
    side: str           # buy | sell
    volume: float
    open_price: float
    close_price: float
    commission: float
    swap: float
    profit: float

    @property
    def net_pnl(self) -> float:
        return self.profit + self.commission + self.swap


def read_html(path: Path) -> str:
    """MT5 exports as UTF-16-LE with BOM. Fall back to UTF-8 if no BOM."""
    raw = path.read_bytes()
    if raw[:2] == b"\xff\xfe":
        return raw.decode("utf-16")
    if raw[:2] == b"\xfe\xff":
        return raw.decode("utf-16-be")
    return raw.decode("utf-8", errors="replace")


def parse_number(s: str) -> float:
    """MT5 uses thousands-separator with non-breaking spaces or regular spaces."""
    return float(s.replace("\xa0", "").replace(" ", "").replace(",", "."))


def parse(html: str) -> tuple[float, list[Trade], dict[str, str]]:
    """Return (initial_balance, trades, summary_metrics)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    rows = soup.find_all("tr")

    initial_balance = None
    trades: list[Trade] = []
    summary: dict[str, str] = {}

    current_section = None
    for tr in rows:
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if not cells:
            continue
        text = " ".join(cells).lower()

        # Section markers
        if "positions" in text and len(cells) == 1:
            current_section = "positions"
            continue
        if "orders" in text and len(cells) == 1:
            current_section = "orders"
            continue
        if cells[0].lower() == "deals" and len(cells) == 1:
            current_section = "deals"
            continue
        if "results" in text and len(cells) == 1:
            current_section = "results"
            continue

        # Initial balance row (Deals section, type='balance' + comment "initial")
        if "balance" in [c.lower() for c in cells] and "initial" in tr.get_text().lower():
            # Comment cell mentions "initial"; the Balance value is the cell
            # immediately before it. (MT5 Deal layout: ... Profit, Balance, Comment.)
            comment_idx = next(
                (i for i, c in enumerate(cells) if "initial" in c.lower()),
                None,
            )
            if comment_idx and comment_idx >= 2:
                try:
                    initial_balance = parse_number(cells[comment_idx - 1])
                except ValueError:
                    pass
            continue

        # Closed positions: 14 cells with cells[3] = buy/sell. There is a
        # hidden <td colspan="8"> at cells[4] (BeautifulSoup sees it as empty
        # text), so the real data is offset by one from what naive indexing
        # would suggest. Layout:
        #   0=OpenTime 1=Position 2=Symbol 3=Type 4=(hidden) 5=Volume
        #   6=OpenPrice 7=S/L 8=T/P 9=CloseTime 10=ClosePrice
        #   11=Commission 12=Swap 13=Profit
        if current_section == "positions" and len(cells) >= 14:
            if cells[3].lower() in {"buy", "sell"}:
                try:
                    close_time = datetime.strptime(cells[9], "%Y.%m.%d %H:%M:%S")
                    trades.append(
                        Trade(
                            close_time=close_time,
                            symbol=cells[2],
                            side=cells[3].lower(),
                            volume=parse_number(cells[5]),
                            open_price=parse_number(cells[6]),
                            close_price=parse_number(cells[10]),
                            commission=parse_number(cells[11]),
                            swap=parse_number(cells[12]),
                            profit=parse_number(cells[13]),
                        )
                    )
                except (ValueError, IndexError):
                    pass
            continue

        # Summary metrics (Results section): pairs of label: value
        if current_section == "results" and len(cells) >= 2:
            # MT5 packs label / value pairs into the same row sometimes
            for i in range(0, len(cells) - 1, 2):
                label = cells[i].rstrip(":").strip()
                value = cells[i + 1].strip()
                if label and value and len(label) < 60:
                    summary.setdefault(label, value)

    if initial_balance is None:
        raise ValueError("No pude encontrar el balance inicial en la sección Deals")
    trades.sort(key=lambda t: t.close_time)
    return initial_balance, trades, summary


def daily_breakdown(initial_balance: float, trades: list[Trade]) -> list[dict]:
    """Group trades by close date and compute running balance + daily DD."""
    by_day: dict[date, list[Trade]] = defaultdict(list)
    for t in trades:
        by_day[t.close_time.date()].append(t)

    rows = []
    running = initial_balance
    peak = initial_balance
    for day in sorted(by_day):
        day_trades = by_day[day]
        open_bal = running
        day_pnl = sum(t.net_pnl for t in day_trades)
        running = open_bal + day_pnl
        peak = max(peak, running)
        daily_pct = (day_pnl / open_bal) * 100 if open_bal else 0.0
        total_dd_pct = ((peak - running) / peak) * 100 if peak else 0.0
        rows.append(
            {
                "date": day,
                "n_trades": len(day_trades),
                "open_balance": open_bal,
                "day_pnl": day_pnl,
                "close_balance": running,
                "daily_pct": daily_pct,
                "running_peak": peak,
                "total_dd_pct": total_dd_pct,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Ruta al MT5 ReportHistory.html")
    parser.add_argument(
        "--daily-limit",
        type=float,
        default=3.0,
        help="Límite de daily loss en %% (default 3 — aplica a instant y 1-step)",
    )
    parser.add_argument(
        "--total-limit",
        type=float,
        default=5.0,
        help="Límite de pérdida total en %% (default 5 — ajustá según las reglas exactas)",
    )
    args = parser.parse_args()

    if not args.report.is_file():
        sys.exit(f"No existe el archivo: {args.report}")

    html = read_html(args.report)
    initial_balance, trades, summary = parse(html)

    print(f"=== {args.report.name} ===\n")
    print(f"Balance inicial:    ${initial_balance:>12,.2f}")
    print(f"Trades cerrados:    {len(trades):>13}")
    if trades:
        print(f"Primer trade:       {trades[0].close_time:%Y-%m-%d %H:%M}")
        print(f"Último trade:       {trades[-1].close_time:%Y-%m-%d %H:%M}")

    rows = daily_breakdown(initial_balance, trades)

    # Tabla día a día
    print("\nDía         #tr    open$        P&L$    close$    daily%   peak$   totalDD%")
    print("─" * 82)
    worst_daily = None
    worst_total = None
    for r in rows:
        flag = ""
        if abs(r["daily_pct"]) >= args.daily_limit and r["daily_pct"] < 0:
            flag += " ⚠DAILY"
        if r["total_dd_pct"] >= args.total_limit:
            flag += " ⚠TOTAL"
        print(
            f"{r['date']}  {r['n_trades']:>3}  "
            f"{r['open_balance']:>9,.2f}  {r['day_pnl']:>+9,.2f}  "
            f"{r['close_balance']:>9,.2f}  {r['daily_pct']:>+6.2f}%  "
            f"{r['running_peak']:>9,.2f}  {r['total_dd_pct']:>5.2f}%{flag}"
        )
        if r["daily_pct"] < 0 and (worst_daily is None or r["daily_pct"] < worst_daily["daily_pct"]):
            worst_daily = r
        if worst_total is None or r["total_dd_pct"] > worst_total["total_dd_pct"]:
            worst_total = r

    print()
    print("─" * 82)
    if worst_daily:
        print(
            f"Peor daily DD:   {worst_daily['daily_pct']:+.2f}% el "
            f"{worst_daily['date']}  (${worst_daily['day_pnl']:+,.2f})"
            f"  límite: -{args.daily_limit}%   "
            + ("✗ BREACH" if abs(worst_daily['daily_pct']) >= args.daily_limit else "✓ OK")
        )
    if worst_total:
        print(
            f"Peor total DD:   {worst_total['total_dd_pct']:.2f}% al "
            f"{worst_total['date']}  límite: {args.total_limit}%   "
            + ("✗ BREACH" if worst_total["total_dd_pct"] >= args.total_limit else "✓ OK")
        )
    final_balance = rows[-1]["close_balance"] if rows else initial_balance
    print(f"Balance final:   ${final_balance:>12,.2f}  (P&L total: ${final_balance - initial_balance:+,.2f})")

    # Reconciliar con el summary del propio MT5
    if "Total Net Profit" in summary:
        reported = parse_number(summary["Total Net Profit"])
        calculated = final_balance - initial_balance
        diff = abs(reported - calculated)
        match = "✓" if diff < 0.5 else "✗"
        print(f"\nReconciliación:  MT5 reporta Net Profit ${reported:+,.2f}  /  calculado ${calculated:+,.2f}  {match}")

    print("\nSummary MT5 (extracto):")
    for k in [
        "Total Net Profit", "Profit Factor", "Balance Drawdown Maximal",
        "Total Trades", "Profit Trades (% of total)", "Loss Trades (% of total)",
        "Maximum consecutive losses ($)",
    ]:
        if k in summary:
            print(f"  {k:32s} {summary[k]}")


if __name__ == "__main__":
    main()
