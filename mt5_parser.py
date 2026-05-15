"""Parser de MetaTrader 5 ReportHistory.html → (initial_balance, trades, summary).

UTF-16 con BOM. La sección Positions tiene rows con un <td colspan=8 hidden>
en posición 4 que desplaza el resto de las columnas.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rules_engine import Trade


def read_html(path: Path) -> str:
    """MT5 exporta UTF-16-LE con BOM. Caer a UTF-8 si no hay BOM."""
    raw = path.read_bytes()
    if raw[:2] == b"\xff\xfe":
        return raw.decode("utf-16")
    if raw[:2] == b"\xfe\xff":
        return raw.decode("utf-16-be")
    return raw.decode("utf-8", errors="replace")


def parse_number(s: str) -> float:
    """MT5 usa NBSP o espacios como thousands-separator."""
    return float(s.replace("\xa0", "").replace(" ", "").replace(",", "."))


def parse(html: str) -> tuple[float, list[Trade], dict[str, str]]:
    """Devuelve (initial_balance, trades, summary_dict)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")

    initial_balance: float | None = None
    trades: list[Trade] = []
    summary: dict[str, str] = {}
    current_section: str | None = None

    for tr in soup.find_all("tr"):
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if not cells:
            continue
        text_lower = " ".join(cells).lower()

        # Section markers
        if len(cells) == 1:
            t = cells[0].lower()
            if t == "positions":
                current_section = "positions"
                continue
            if t == "orders":
                current_section = "orders"
                continue
            if t == "deals":
                current_section = "deals"
                continue
            if t == "results":
                current_section = "results"
                continue

        # Initial balance: fila type='balance' con comment "initial"
        if "balance" in [c.lower() for c in cells] and "initial" in tr.get_text().lower():
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

        # Closed positions — 14 cells, cells[3] = buy/sell, cells[4] = hidden vacío
        # Layout: 0=OpenTime 1=Position 2=Symbol 3=Type 4=(hidden) 5=Volume
        #         6=OpenPrice 7=S/L 8=T/P 9=CloseTime 10=ClosePrice
        #         11=Commission 12=Swap 13=Profit
        if current_section == "positions" and len(cells) >= 14:
            if cells[3].lower() in {"buy", "sell"}:
                try:
                    open_time = datetime.strptime(cells[0], "%Y.%m.%d %H:%M:%S")
                    close_time = datetime.strptime(cells[9], "%Y.%m.%d %H:%M:%S")
                    # S/L y T/P pueden estar vacíos ('') o '0' cuando no se setearon
                    def _opt(s: str) -> float:
                        if not s or s.strip() in {"", "0", "0.00"}:
                            return 0.0
                        try:
                            return parse_number(s)
                        except ValueError:
                            return 0.0
                    trades.append(Trade(
                        open_time=open_time,
                        close_time=close_time,
                        symbol=cells[2],
                        side=cells[3].lower(),
                        volume=parse_number(cells[5]),
                        open_price=parse_number(cells[6]),
                        stop_loss=_opt(cells[7]),
                        take_profit=_opt(cells[8]),
                        close_price=parse_number(cells[10]),
                        commission=parse_number(cells[11]),
                        swap=parse_number(cells[12]),
                        profit=parse_number(cells[13]),
                    ))
                except (ValueError, IndexError):
                    pass
            continue

        # Summary: pares label / value
        if current_section == "results" and len(cells) >= 2:
            for i in range(0, len(cells) - 1, 2):
                label = cells[i].rstrip(":").strip()
                value = cells[i + 1].strip()
                if label and value and len(label) < 60:
                    summary.setdefault(label, value)

    if initial_balance is None:
        raise ValueError("No pude encontrar el balance inicial en la sección Deals")
    trades.sort(key=lambda t: t.close_time)
    return initial_balance, trades, summary
