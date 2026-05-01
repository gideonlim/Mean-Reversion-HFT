"""Generate a daily strategy report from trade_log.json and live Alpaca data.

Outputs:
- Markdown report (to stdout + GHA step summary)
- Equity curve PNG (logs/report_equity.png)
- Email with HTML report + chart attachment (if EMAIL_ADDRESS and EMAIL_APP_PASSWORD are set)
"""
from __future__ import annotations

import csv
import json
import math
import mimetypes
import os
import smtplib
import sys
from datetime import date, timedelta
from email.message import EmailMessage
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest
from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import SETTINGS, et_now, et_today

LOG_FILE = Path(SETTINGS.LOG_DIR) / "trade_log.json"
REPORT_DIR = Path(SETTINGS.REPORT_DIR)


def _report_paths(et_date_iso: str) -> tuple[Path, Path, Path]:
    """Return (chart_path, pdf_path, csv_path) for a given ET date.

    Files are date-stamped so daily reports accumulate as a history.
    """
    return (
        REPORT_DIR / f"equity_{et_date_iso}.png",
        REPORT_DIR / f"report_{et_date_iso}.pdf",
        REPORT_DIR / f"trades_{et_date_iso}.csv",
    )


def _enum_value(e) -> str:
    """Robustly extract a string from an Alpaca enum."""
    if hasattr(e, "value"):
        return str(e.value).lower()
    return str(e).split(".")[-1].lower()


def load_log() -> list[dict]:
    if LOG_FILE.exists():
        try:
            return json.loads(LOG_FILE.read_text())
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def build_equity_series(entries: list[dict]) -> list[tuple[date, float]]:
    """Extract one equity reading per ET date (use 'post' if available, else 'pre')."""
    by_date: dict[str, float] = {}
    for e in entries:
        d = e["et_date"]
        eq = e["account"]["equity"]
        if e["label"] == "post":
            by_date[d] = eq
        elif d not in by_date:
            by_date[d] = eq
    return sorted(
        [(date.fromisoformat(d), eq) for d, eq in by_date.items()],
        key=lambda x: x[0],
    )


def compute_report_stats(
    equity_series: list[tuple[date, float]],
    current_equity: float,
    positions: list[dict],
) -> dict:
    """Compute key stats for the report."""
    starting_capital = SETTINGS.STARTING_CAPITAL
    if not equity_series:
        return {}

    equities = [eq for _, eq in equity_series]
    dates = [d for d, _ in equity_series]

    total_return = (current_equity / starting_capital - 1) * 100
    n_days = (dates[-1] - dates[0]).days if len(dates) > 1 else 1

    # Annualized stats only meaningful with enough history; otherwise show N/A.
    has_enough_history = n_days >= SETTINGS.MIN_DAYS_FOR_ANNUALIZATION

    if has_enough_history:
        ann_factor = 365.25 / n_days
        annualized_return = ((current_equity / starting_capital) ** ann_factor - 1) * 100
    else:
        annualized_return = None

    # Daily log returns for Sharpe
    daily_returns = []
    for i in range(1, len(equities)):
        if equities[i - 1] > 0:
            daily_returns.append(math.log(equities[i] / equities[i - 1]))

    if len(daily_returns) > 1 and has_enough_history:
        mu = np.mean(daily_returns)
        sigma = np.std(daily_returns, ddof=1)
        daily_sharpe = mu / sigma if sigma > 1e-12 else 0.0
        ann_sharpe = daily_sharpe * math.sqrt(252)
    else:
        daily_sharpe = None
        ann_sharpe = None

    # Drawdown — peak starts at MAX(starting_capital, first reading) so any
    # early loss is correctly measured against the true high-water mark.
    peak = max(starting_capital, equities[0])
    max_dd = 0.0
    for eq in equities:
        peak = max(peak, eq)
        dd = (eq / peak - 1) * 100
        max_dd = min(max_dd, dd)

    # Position summary
    pos_str = "flat"
    if positions:
        p = positions[0]
        pos_str = f"{p['qty']} {p['symbol']} {p['side']} @ ${p['avg_entry_price']:.2f}"

    return {
        "current_equity": current_equity,
        "total_return_pct": total_return,
        "total_pnl": current_equity - starting_capital,
        "annualized_return_pct": annualized_return,
        "daily_sharpe": daily_sharpe,
        "annualized_sharpe": ann_sharpe,
        "max_drawdown_pct": max_dd,
        "trading_days": len(equities),
        "calendar_days": n_days,
        "position": pos_str,
        "starting_capital": starting_capital,
        "has_enough_history": has_enough_history,
        "min_days_for_annualization": SETTINGS.MIN_DAYS_FOR_ANNUALIZATION,
    }


def plot_equity(
    equity_series: list[tuple[date, float]],
    current_equity: float,
    out_path: Path,
) -> Path:
    """Plot equity curve and save to PNG at out_path."""
    dates = [d for d, _ in equity_series]
    equities = [eq for _, eq in equity_series]

    # Append today if not already in series
    today = et_today()
    if not dates or dates[-1] < today:
        dates.append(today)
        equities.append(current_equity)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, equities, linewidth=1.8, color="#2563eb")
    ax.axhline(SETTINGS.STARTING_CAPITAL, color="grey", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.fill_between(dates, SETTINGS.STARTING_CAPITAL, equities, alpha=0.08, color="#2563eb")

    ax.set_title(f"{SETTINGS.SYMBOL} Mean-Reversion — Account Equity", fontsize=13, fontweight="bold")
    ax.set_ylabel("Equity ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _fmt_pct(value: float | None) -> str:
    return f"{value:+.2f}%" if value is not None else f"N/A (need >={SETTINGS.MIN_DAYS_FOR_ANNUALIZATION} days)"


def _fmt_sharpe(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else f"N/A (need >={SETTINGS.MIN_DAYS_FOR_ANNUALIZATION} days)"


def format_markdown(stats: dict, recent_orders: list[dict]) -> str:
    """Build the markdown report."""
    if not stats:
        return "**No data available yet.** Run backtest or wait for first trade.\n"

    pnl_sign = "+" if stats["total_pnl"] >= 0 else ""

    orders_rows = ""
    for o in recent_orders[-5:]:
        fill = f"${o['filled_avg_price']:.2f}" if o.get("filled_avg_price") else "—"
        orders_rows += f"| {o['created_at'][:10]} | {o['side'].upper()} {o['filled_qty']}/{o['qty']} | {o['status']} | {fill} |\n"
    if not orders_rows:
        orders_rows = "| — | — | — | — |\n"

    return f"""## Daily Report — {et_today().isoformat()}

### Account
| | |
|---|---|
| **Equity** | ${stats['current_equity']:,.2f} |
| **Starting Capital** | ${stats['starting_capital']:,.2f} |
| **Total P&L** | {pnl_sign}${stats['total_pnl']:,.2f} ({pnl_sign}{stats['total_return_pct']:.2f}%) |
| **Annualized Return** | {_fmt_pct(stats['annualized_return_pct'])} |
| **Position** | {stats['position']} |

### Risk
| | |
|---|---|
| **Daily Sharpe** | {_fmt_sharpe(stats['daily_sharpe'])} |
| **Annualized Sharpe** | {_fmt_sharpe(stats['annualized_sharpe'])} |
| **Max Drawdown** | {stats['max_drawdown_pct']:.2f}% |
| **Trading Days** | {stats['trading_days']} ({stats['calendar_days']} calendar days) |

### Recent Orders
| Date | Side | Status | Fill Price |
|---|---|---|---|
{orders_rows}
"""


def fetch_all_orders(client: TradingClient) -> list:
    """Fetch all orders for SETTINGS.SYMBOL since beginning of period (paginated)."""
    all_orders = []
    seen_ids = set()
    until = None
    page_size = 500
    while True:
        kwargs = {
            "status": QueryOrderStatus.ALL,
            "symbols": [SETTINGS.SYMBOL],
            "limit": page_size,
            "direction": "desc",
        }
        if until is not None:
            kwargs["until"] = until
        page = client.get_orders(filter=GetOrdersRequest(**kwargs))
        if not page:
            break
        new = [o for o in page if o.id not in seen_ids]
        if not new:
            break
        all_orders.extend(new)
        seen_ids.update(o.id for o in new)
        if len(page) < page_size:
            break
        until = min(o.created_at for o in page)
    return sorted(all_orders, key=lambda x: x.created_at)


def write_trades_csv(orders: list, path: Path) -> Path:
    """Write all orders to a CSV file."""
    path.parent.mkdir(exist_ok=True)
    fields = [
        "created_at", "submitted_at", "filled_at", "symbol", "side", "qty",
        "filled_qty", "time_in_force", "type", "status", "limit_price",
        "filled_avg_price", "client_order_id", "id",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for o in orders:
            w.writerow({
                "created_at": str(o.created_at)[:19] if o.created_at else "",
                "submitted_at": str(o.submitted_at)[:19] if o.submitted_at else "",
                "filled_at": str(o.filled_at)[:19] if o.filled_at else "",
                "symbol": o.symbol,
                "side": _enum_value(o.side),
                "qty": o.qty,
                "filled_qty": o.filled_qty if o.filled_qty else "",
                "time_in_force": _enum_value(o.time_in_force),
                "type": _enum_value(o.order_type) if hasattr(o, "order_type") else "",
                "status": _enum_value(o.status),
                "limit_price": o.limit_price if o.limit_price else "",
                "filled_avg_price": o.filled_avg_price if o.filled_avg_price else "",
                "client_order_id": o.client_order_id,
                "id": str(o.id),
            })
    return path


def generate_pdf(
    stats: dict,
    recent_orders: list[dict],
    chart_path: Path,
    out_path: Path,
) -> Path:
    """Render a PDF report using reportlab."""
    out_path.parent.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=f"{SETTINGS.SYMBOL} Mean-Reversion Report",
    )
    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], spaceBefore=12, spaceAfter=6)
    body = styles["BodyText"]
    muted = ParagraphStyle("muted", parent=body, textColor=colors.HexColor("#6b7280"), fontSize=9)

    story = []
    story.append(Paragraph(f"{SETTINGS.SYMBOL} Mean-Reversion Report", h1))
    story.append(Paragraph(
        f"{et_today().isoformat()} &middot; {SETTINGS.SYMBOL} &middot; "
        f"Long-only: {SETTINGS.LONG_ONLY}",
        muted,
    ))
    story.append(Spacer(1, 0.4 * cm))

    if not stats:
        story.append(Paragraph("No data available yet.", body))
        doc.build(story)
        return out_path

    pnl_sign = "+" if stats["total_pnl"] >= 0 else ""
    pnl_color = colors.HexColor("#16a34a") if stats["total_pnl"] >= 0 else colors.HexColor("#dc2626")

    # Account section
    story.append(Paragraph("Account", h2))
    account_rows = [
        ["Equity", f"${stats['current_equity']:,.2f}"],
        ["Starting Capital", f"${stats['starting_capital']:,.2f}"],
        ["Total P&L", f"{pnl_sign}${stats['total_pnl']:,.2f}  ({pnl_sign}{stats['total_return_pct']:.2f}%)"],
        ["Annualized Return", _fmt_pct(stats["annualized_return_pct"])],
        ["Position", stats["position"]],
    ]
    story.append(_kv_table(account_rows, pnl_color_idx=2 if stats["total_pnl"] != 0 else None, pnl_color=pnl_color))

    # Risk section
    story.append(Paragraph("Risk", h2))
    risk_rows = [
        ["Daily Sharpe", _fmt_sharpe(stats["daily_sharpe"])],
        ["Annualized Sharpe", _fmt_sharpe(stats["annualized_sharpe"])],
        ["Max Drawdown", f"{stats['max_drawdown_pct']:.2f}%"],
        ["Trading Days", f"{stats['trading_days']}  ({stats['calendar_days']} calendar days)"],
    ]
    story.append(_kv_table(risk_rows))

    # Equity chart
    if chart_path.exists():
        story.append(Paragraph("Equity Curve", h2))
        story.append(Image(str(chart_path), width=16 * cm, height=7 * cm))

    # Recent orders
    story.append(Paragraph("Recent Orders", h2))
    order_rows = [["Date", "Side", "Qty (filled / total)", "Status", "Fill Price"]]
    for o in recent_orders[-10:]:
        fill = f"${o['filled_avg_price']:.2f}" if o.get("filled_avg_price") else "—"
        order_rows.append([
            o["created_at"][:10],
            o["side"].upper(),
            f"{o['filled_qty']} / {o['qty']}",
            o["status"],
            fill,
        ])
    if len(order_rows) == 1:
        order_rows.append(["—", "—", "—", "—", "—"])

    order_tbl = Table(order_rows, colWidths=[2.5 * cm, 2 * cm, 4 * cm, 3 * cm, 3 * cm])
    order_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f9fafb")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor("#e5e7eb")),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(order_tbl)

    doc.build(story)
    return out_path


def _kv_table(rows: list[list[str]], pnl_color_idx: int | None = None, pnl_color=None) -> Table:
    """Helper: render a 2-column key/value table for the PDF."""
    tbl = Table(rows, colWidths=[5 * cm, 11 * cm])
    style = [
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#6b7280")),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if pnl_color_idx is not None and pnl_color is not None:
        style.append(("TEXTCOLOR", (1, pnl_color_idx), (1, pnl_color_idx), pnl_color))
    tbl.setStyle(TableStyle(style))
    return tbl


def main() -> int:
    load_dotenv()
    api_key = os.environ.get("APCA_API_KEY_ID")
    api_secret = os.environ.get("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        print("ERROR: APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set", file=sys.stderr)
        return 2

    client = TradingClient(api_key, api_secret, paper=True)
    acct = client.get_account()
    current_equity = float(acct.equity)

    # Current positions
    positions = []
    for p in client.get_all_positions():
        positions.append({
            "symbol": p.symbol,
            "side": _enum_value(p.side),
            "qty": int(float(p.qty)),
            "avg_entry_price": float(p.avg_entry_price),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
        })

    # Recent orders
    since = (et_now() - timedelta(days=7)).isoformat()
    try:
        orders = client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            after=since,
            symbols=[SETTINGS.SYMBOL],
            limit=10,
        ))
        recent_orders = sorted(
            [
                {
                    "created_at": str(o.created_at)[:19],
                    "side": _enum_value(o.side),
                    "qty": int(float(o.qty)),
                    "filled_qty": int(float(o.filled_qty)) if o.filled_qty else 0,
                    "status": _enum_value(o.status),
                    "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                }
            for o in orders],
            key=lambda x: x["created_at"],
        )
    except Exception:
        recent_orders = []

    # Build from trade log
    entries = load_log()
    equity_series = build_equity_series(entries)
    stats = compute_report_stats(equity_series, current_equity, positions)

    # Generate outputs
    md = format_markdown(stats, recent_orders)
    print(md)

    # Date-stamped output paths in the report directory
    et_date_iso = et_today().isoformat()
    chart_path, pdf_path, csv_path = _report_paths(et_date_iso)

    chart_path = plot_equity(equity_series, current_equity, chart_path)
    print(f"Equity chart saved to {chart_path}")

    # PDF report
    pdf_path = generate_pdf(stats, recent_orders, chart_path, pdf_path)
    print(f"PDF report saved to {pdf_path}")

    # CSV of all trades since beginning of period
    try:
        all_orders = fetch_all_orders(client)
        csv_path = write_trades_csv(all_orders, csv_path)
        print(f"Trades CSV saved to {csv_path} ({len(all_orders)} orders)")
    except Exception as e:
        print(f"Failed to fetch full order history: {e}", file=sys.stderr)
        csv_path = None

    # GHA step summary
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a") as f:
                f.write(md)
        except OSError:
            pass

    # Email report with PDF + CSV attached
    send_email_report(stats, recent_orders, chart_path, pdf_path, csv_path)

    return 0


def send_email_report(
    stats: dict,
    recent_orders: list[dict],
    chart_path: Path,
    pdf_path: Path | None = None,
    csv_path: Path | None = None,
) -> None:
    """Send the report as an HTML email with the equity chart attached via Gmail SMTP."""
    email_addr = os.environ.get("EMAIL_ADDRESS")
    email_pass = os.environ.get("EMAIL_APP_PASSWORD")
    if not email_addr or not email_pass:
        print("EMAIL_ADDRESS / EMAIL_APP_PASSWORD not set — skipping email")
        return

    pnl_sign = "+" if stats.get("total_pnl", 0) >= 0 else ""
    pnl_color = "#16a34a" if stats.get("total_pnl", 0) >= 0 else "#dc2626"

    orders_html = ""
    for o in recent_orders[-5:]:
        fill = f"${o['filled_avg_price']:.2f}" if o.get("filled_avg_price") else "—"
        orders_html += f"<tr><td>{o['created_at'][:10]}</td><td>{o['side'].upper()} {o['filled_qty']}/{o['qty']}</td><td>{o['status']}</td><td>{fill}</td></tr>\n"
    if not orders_html:
        orders_html = "<tr><td colspan='4'>No recent orders</td></tr>"

    html = f"""\
<html><body style="font-family: -apple-system, Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
<h2 style="margin-bottom: 4px;">Mean-Reversion Daily Report</h2>
<p style="color: #6b7280; margin-top: 0;">{et_today().isoformat()} &middot; {SETTINGS.SYMBOL}</p>

<table style="border-collapse: collapse; width: 100%; margin: 16px 0;">
<tr><td style="padding: 8px; border-bottom: 1px solid #e5e7eb; color: #6b7280;">Equity</td>
    <td style="padding: 8px; border-bottom: 1px solid #e5e7eb; font-weight: bold; text-align: right;">${stats.get('current_equity', 0):,.2f}</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #e5e7eb; color: #6b7280;">Total P&L</td>
    <td style="padding: 8px; border-bottom: 1px solid #e5e7eb; font-weight: bold; text-align: right; color: {pnl_color};">{pnl_sign}${stats.get('total_pnl', 0):,.2f} ({pnl_sign}{stats.get('total_return_pct', 0):.2f}%)</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #e5e7eb; color: #6b7280;">Position</td>
    <td style="padding: 8px; border-bottom: 1px solid #e5e7eb; text-align: right;">{stats.get('position', 'flat')}</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #e5e7eb; color: #6b7280;">Ann. Sharpe</td>
    <td style="padding: 8px; border-bottom: 1px solid #e5e7eb; text-align: right;">{_fmt_sharpe(stats.get('annualized_sharpe'))}</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #e5e7eb; color: #6b7280;">Max Drawdown</td>
    <td style="padding: 8px; border-bottom: 1px solid #e5e7eb; text-align: right;">{stats.get('max_drawdown_pct', 0):.2f}%</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #e5e7eb; color: #6b7280;">Trading Days</td>
    <td style="padding: 8px; border-bottom: 1px solid #e5e7eb; text-align: right;">{stats.get('trading_days', 0)}</td></tr>
</table>

<h3>Equity Curve</h3>
<img src="cid:equity_chart" style="width: 100%; max-width: 580px;" alt="Equity curve" />

<h3 style="margin-top: 24px;">Recent Orders</h3>
<table style="border-collapse: collapse; width: 100%; font-size: 14px;">
<tr style="background: #f9fafb;">
  <th style="padding: 6px 8px; text-align: left; border-bottom: 2px solid #e5e7eb;">Date</th>
  <th style="padding: 6px 8px; text-align: left; border-bottom: 2px solid #e5e7eb;">Side</th>
  <th style="padding: 6px 8px; text-align: left; border-bottom: 2px solid #e5e7eb;">Status</th>
  <th style="padding: 6px 8px; text-align: left; border-bottom: 2px solid #e5e7eb;">Fill</th>
</tr>
{orders_html}
</table>

<p style="color: #9ca3af; font-size: 12px; margin-top: 24px;">
  Starting capital: ${stats.get('starting_capital', 0):,.2f} &middot;
  Ann. return: {_fmt_pct(stats.get('annualized_return_pct'))} &middot;
  Daily Sharpe: {_fmt_sharpe(stats.get('daily_sharpe'))}
</p>
</body></html>"""

    msg = EmailMessage()
    msg["Subject"] = (
        f"{SETTINGS.SYMBOL} Report {et_today().isoformat()} — "
        f"{pnl_sign}${stats.get('total_pnl', 0):,.2f} ({pnl_sign}{stats.get('total_return_pct', 0):.2f}%)"
    )
    msg["From"] = email_addr
    msg["To"] = email_addr
    msg.set_content("Your email client does not support HTML. View in a browser.")
    msg.add_alternative(html, subtype="html")

    if chart_path.exists():
        with open(chart_path, "rb") as img:
            msg.get_payload()[1].add_related(
                img.read(), maintype="image", subtype="png", cid="equity_chart"
            )

    # Attach PDF and CSV
    for path in (pdf_path, csv_path):
        if path is None or not path.exists():
            continue
        ctype, _ = mimetypes.guess_type(str(path))
        maintype, subtype = (ctype or "application/octet-stream").split("/", 1)
        with open(path, "rb") as fh:
            msg.add_attachment(
                fh.read(), maintype=maintype, subtype=subtype, filename=path.name
            )

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(email_addr, email_pass)
            s.send_message(msg)
        print(f"Email sent to {email_addr}")
    except Exception as e:
        print(f"Email failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
