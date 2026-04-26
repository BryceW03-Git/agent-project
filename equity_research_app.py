import streamlit as st
import os
import json
import io
import pandas as pd
from dotenv import load_dotenv
import anthropic
import yfinance as yf
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

load_dotenv()
client = anthropic.Anthropic()

# ─── CONSTANTS ───────────────────────────────────────────────

PERIOD_MAP = {
    "YTD":    "ytd",
    "1 Year": "1y",
    "3 Years":"3y",
    "5 Years":"5y",
}

INDICES = {
    "S&P 500":   "^GSPC",
    "NASDAQ":    "^IXIC",
    "Dow Jones": "^DJI",
}

CHART_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf"
]

# ─── DATA FUNCTIONS ──────────────────────────────────────────

def get_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data  = stock.history(period="5d")
        if data.empty:
            return f"Error: No data found for {ticker}"
        price = data["Close"].iloc[-1]
        date  = data.index[-1].date()
        return f"{ticker} most recent price: ${price:.2f} (as of {date})"
    except Exception as e:
        return f"Error fetching {ticker}: {str(e)}"


def get_historical_return(ticker, years):
    try:
        stock = yf.Ticker(ticker)
        data  = stock.history(period=f"{years + 1}y")
        if data.empty or len(data) < 2:
            return f"Error: Insufficient data for {ticker}"
        start_price  = data["Close"].iloc[0]
        end_price    = data["Close"].iloc[-1]
        total_return = ((end_price - start_price) / start_price) * 100
        start_date   = data.index[0].date()
        end_date     = data.index[-1].date()
        return (f"{ticker} {years}-year return: {total_return:.2f}% "
                f"(${start_price:.2f} to ${end_price:.2f}, "
                f"{start_date} to {end_date})")
    except Exception as e:
        return f"Error computing return for {ticker}: {str(e)}"


def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        if (not info
                or (info.get("regularMarketPrice") is None
                    and info.get("currentPrice") is None
                    and info.get("previousClose") is None)):
            return None

        def pct(val):
            return "N/A" if val is None else f"{val * 100:.1f}%"

        def multiple(val, decimals=1):
            return "N/A" if val is None else f"{val:.{decimals}f}x"

        def dollar_b(val):
            return "N/A" if val is None else f"${val / 1e9:.1f}B"

        market_cap       = info.get("marketCap")
        enterprise_value = info.get("enterpriseValue")
        ebitda           = info.get("ebitda")
        free_cashflow    = info.get("freeCashflow")

        ev_ebitda    = None
        price_to_fcf = None
        fcf_yield    = None

        if enterprise_value and ebitda and ebitda > 0:
            ev_ebitda = enterprise_value / ebitda
        if market_cap and free_cashflow and free_cashflow > 0:
            price_to_fcf = market_cap / free_cashflow
        if market_cap and free_cashflow and market_cap > 0:
            fcf_yield = free_cashflow / market_cap

        return {
            "company_name":       info.get("longName", ticker),
            "sector":             info.get("sector", "N/A"),
            "industry":           info.get("industry", "N/A"),
            "market_cap":         dollar_b(market_cap),
            "fifty_two_week_high":f"${info.get('fiftyTwoWeekHigh', 0):.2f}",
            "fifty_two_week_low": f"${info.get('fiftyTwoWeekLow', 0):.2f}",
            "1_trailing_pe":      multiple(info.get("trailingPE")),
            "2_forward_pe":       multiple(info.get("forwardPE")),
            "3_ev_ebitda":        multiple(ev_ebitda),
            "4_price_to_book":    multiple(info.get("priceToBook")),
            "5_price_to_fcf":     multiple(price_to_fcf),
            "6_operating_margin": pct(info.get("operatingMargins")),
            "7_return_on_equity": pct(info.get("returnOnEquity")),
            "8_revenue_growth":   pct(info.get("revenueGrowth")),
            "9_debt_to_equity":   multiple(info.get("debtToEquity"), decimals=2),
            "10_fcf_yield":       pct(fcf_yield),
        }
    except Exception:
        return None


def validate_ticker(ticker):
    result = get_fundamentals(ticker)
    if result is None:
        return False, None
    return True, result.get("company_name", ticker)


def get_comps_data(comp_tickers):
    valid   = {}
    invalid = []
    for ticker in comp_tickers:
        ticker = ticker.upper().strip()
        result = get_fundamentals(ticker)
        if result is not None:
            valid[ticker] = result
        else:
            invalid.append(ticker)
    return valid, invalid


# ─── PLOTLY CHART (interactive) ──────────────────────────────

def build_price_chart(subject_ticker, comp_tickers, period,
                      normalize, selected_indices=None):
    all_series = {}

    def fetch(symbol, label):
        try:
            data = yf.Ticker(symbol).history(period=period)
            if not data.empty:
                all_series[label] = data["Close"]
        except Exception:
            pass

    fetch(subject_ticker, subject_ticker)
    for t in comp_tickers:
        fetch(t, t)
    for name, symbol in INDICES.items():
        fetch(symbol, name)

    if not all_series:
        return None

    if normalize:
        all_series = {k: (v / v.iloc[0] * 100) for k, v in all_series.items()}

    fig = go.Figure()
    for i, (name, series) in enumerate(all_series.items()):
        is_subject = (name == subject_ticker)
        is_index   = (name in INDICES)

        if is_subject or name in comp_tickers:
            visibility = True
        elif is_index:
            visibility = (
                True if (selected_indices and name in selected_indices)
                else "legendonly"
            )
        else:
            visibility = "legendonly"

        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            name=name,
            mode="lines",
            line=dict(
                color=CHART_COLORS[i % len(CHART_COLORS)],
                width=3 if is_subject else 1.5,
                dash="dot" if is_index else "solid"
            ),
            visible=visibility
        ))

    y_label = "Indexed Price (Base = 100)" if normalize else "Price (USD)"
    title   = f"{subject_ticker} — {period}"
    if normalize:
        title += " (Normalized to 100)"

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    )
    return fig


# ─── MATPLOTLIB CHART (for PDF) ──────────────────────────────

def build_pdf_chart(subject_ticker, comp_tickers, period,
                    normalize, selected_indices=None):
    all_series = {}

    def fetch(symbol, label):
        try:
            data = yf.Ticker(symbol).history(period=period)
            if not data.empty:
                all_series[label] = data["Close"]
        except Exception:
            pass

    fetch(subject_ticker, subject_ticker)
    for t in comp_tickers:
        fetch(t, t)
    for name, symbol in INDICES.items():
        if selected_indices and name in selected_indices:
            fetch(symbol, name)

    if not all_series:
        return None

    if normalize:
        all_series = {k: (v / v.iloc[0] * 100) for k, v in all_series.items()}

    fig, ax = plt.subplots(figsize=(7.5, 3.5), dpi=150)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for i, (name, series) in enumerate(all_series.items()):
        is_subject = (name == subject_ticker)
        is_index   = (name in INDICES)

        try:
            x = series.index.tz_localize(None)
        except Exception:
            try:
                x = series.index.tz_convert(None)
            except Exception:
                x = series.index

        ax.plot(
            x, series.values,
            label=name,
            color=CHART_COLORS[i % len(CHART_COLORS)],
            linewidth=2.5 if is_subject else 1.2,
            linestyle="--" if is_index else "-",
            zorder=3 if is_subject else 2
        )

    ax.set_xlabel("Date", fontsize=8, fontfamily="sans-serif")
    y_label = "Indexed Price (Base = 100)" if normalize else "Price (USD)"
    ax.set_ylabel(y_label, fontsize=8, fontfamily="sans-serif")
    title = f"{subject_ticker} Price History — {period}"
    if normalize:
        title += " (Normalized to 100)"
    ax.set_title(title, fontsize=9, fontweight="bold", fontfamily="sans-serif")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, color="#e0e0e0", linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(0.5)

    ax.legend(
        fontsize=7, loc="upper left",
        framealpha=0.9, edgecolor="#cccccc",
        ncol=min(len(all_series), 4)
    )

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ─── PDF GENERATOR ───────────────────────────────────────────

def generate_pdf(report, subject_fund, comps_data,
                 comp_tickers, chart_period, normalize,
                 selected_indices=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.2 * inch,
        rightMargin=0.2 * inch,
        topMargin=0.3 * inch,
        bottomMargin=0.2 * inch,
    )

    W = letter[0] - 0.4 * inch

    def style(name, **kwargs):
        if "fontName" not in kwargs:
            kwargs["fontName"] = "Helvetica"
        return ParagraphStyle(name, **kwargs)

    s_title     = style("title",     fontSize=16,
                        fontName="Helvetica-Bold", spaceAfter=4)
    s_subtitle  = style("subtitle",  fontSize=9,
                        textColor=colors.HexColor("#555555"), spaceAfter=2)
    s_date      = style("date",      fontSize=8,
                        textColor=colors.HexColor("#888888"), spaceAfter=12)
    s_section   = style("section",   fontSize=10,
                        fontName="Helvetica-Bold",
                        spaceBefore=6, spaceAfter=4,
                        textColor=colors.HexColor("#111111"))
    s_body      = style("body",      fontSize=9, leading=14,
                        spaceAfter=4, alignment=TA_JUSTIFY)
    s_small     = style("small",     fontSize=7.5,
                        textColor=colors.HexColor("#666666"),
                        leading=11, spaceAfter=2, alignment=TA_JUSTIFY)
    s_rating    = style("rating",    fontSize=11,
                        fontName="Helvetica-Bold",
                        textColor=colors.white, alignment=TA_CENTER)
    s_oneliner  = style("oneliner",  fontSize=9,
                        fontName="Helvetica-Bold",
                        textColor=colors.HexColor("#222222"), spaceAfter=2)
    s_rationale = style("rationale", fontSize=8,
                        textColor=colors.HexColor("#444444"),
                        spaceAfter=8, alignment=TA_JUSTIFY)

    def header_table_style(n_rows):
        return TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#111111")),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, 0), 8),
            ("ALIGN",        (0, 0), (-1, 0), "CENTER"),
            ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",     (0, 1), (-1, -1), 8),
            ("ALIGN",        (0, 1), (0, -1),  "LEFT"),
            ("ALIGN",        (1, 1), (-1, -1), "CENTER"),
            *[("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f7f7f7"))
              for i in range(2, n_rows, 2)],
            ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ])

    def perf_table_style():
        return TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#111111")),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, 0), 7.5),
            ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME",     (0, 1), (-1, 1), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 1), (-1, 1), 10),
            ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ])

    story = []

    # ── Header ────────────────────────────────────────────────
    rating_bg = {
        "Strong Buy":  "#1a7f3c",
        "Buy":         "#2e9e57",
        "Hold":        "#b8860b",
        "Sell":        "#c0392b",
        "Strong Sell": "#922b21",
    }.get(report['ai_rating'], "#444444")

    rating_tbl = Table(
        [[Paragraph(f"AI RATING: {report['ai_rating'].upper()}", s_rating)]],
        colWidths=[2.5 * inch]
    )
    rating_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor(rating_bg)),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(KeepTogether([
        Paragraph(f"{report['company_name']} ({report['ticker']})", s_title),
        Paragraph(
            f"{report['sector']} &nbsp;|&nbsp; {report['industry']}", s_subtitle
        ),
        Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y')} &nbsp;|&nbsp; "
            f"AI-Generated Equity Research — Not Investment Advice", s_date
        ),
        HRFlowable(width=W, thickness=1.5,
                   color=colors.HexColor("#111111"), spaceAfter=8),
        rating_tbl,
        Spacer(1, 6),
        Paragraph(report['one_line_summary'], s_oneliner),
        Paragraph(
            f"<i>Rationale: {report['ai_rating_rationale']}</i>", s_rationale
        ),
        HRFlowable(width=W, thickness=0.3,
                   color=colors.HexColor("#cccccc"), spaceAfter=6),
    ]))

    # ── Performance ───────────────────────────────────────────
    perf_headers = [
        "Current Price", "1-Year Return",
        "5-Year Return", "Market Cap", "52-Week Range"
    ]
    perf_values = [
        f"${report['current_price']:.2f}",
        f"{report['one_year_return_pct']:.1f}%",
        f"{report['five_year_return_pct']:.1f}%",
        report['market_cap'],
        f"{report['fifty_two_week_low']} – {report['fifty_two_week_high']}"
    ]
    col_w    = W / len(perf_headers)
    perf_tbl = Table(
        [perf_headers, perf_values],
        colWidths=[col_w] * len(perf_headers)
    )
    perf_tbl.setStyle(perf_table_style())

    story.append(KeepTogether([
        Paragraph("PRICE &amp; PERFORMANCE", s_section),
        perf_tbl,
        Spacer(1, 8),
        HRFlowable(width=W, thickness=0.3,
                   color=colors.HexColor("#cccccc"), spaceAfter=4),
    ]))

    # ── Fundamentals ──────────────────────────────────────────
    val_rows = [
        ["1. Trailing P/E",   report['trailing_pe']],
        ["2. Forward P/E",    report['forward_pe']],
        ["3. EV / EBITDA",    report['ev_ebitda']],
        ["4. Price / Book",   report['price_to_book']],
        ["5. Price / FCF",    report['price_to_fcf']],
    ]
    qual_rows = [
        ["6. Operating Margin", report['operating_margin']],
        ["7. Return on Equity", report['return_on_equity']],
        ["8. Revenue Growth",   report['revenue_growth_yoy']],
        ["9. Debt / Equity",    report['debt_to_equity']],
        ["10. FCF Yield",       report['fcf_yield']],
    ]

    def make_fund_table(rows, col_widths):
        data = [["Metric", report['ticker']]] + rows
        tbl  = Table(data, colWidths=col_widths)
        tbl.setStyle(header_table_style(len(data)))
        return tbl

    half     = W / 2 - 0.1 * inch
    val_tbl  = make_fund_table(val_rows,  [half * 0.65, half * 0.35])
    qual_tbl = make_fund_table(qual_rows, [half * 0.65, half * 0.35])

    fund_layout = Table(
        [[val_tbl, qual_tbl]],
        colWidths=[half, half], hAlign="LEFT"
    )
    fund_layout.setStyle(TableStyle([
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",(0, 0), (-1, -1), 8),
    ]))

    story.append(KeepTogether([
        Paragraph("TOP 10 VALUE INVESTOR FUNDAMENTALS", s_section),
        fund_layout,
        Spacer(1, 8),
        HRFlowable(width=W, thickness=0.3,
                   color=colors.HexColor("#cccccc"), spaceAfter=4),
    ]))

    # ── Chart ─────────────────────────────────────────────────
    period_code = PERIOD_MAP[chart_period]
    chart_bytes = build_pdf_chart(
        report['ticker'], comp_tickers, period_code,
        normalize, selected_indices
    )

    chart_elements = [Paragraph("PRICE HISTORY", s_section)]
    if chart_bytes:
        chart_buf = io.BytesIO(chart_bytes)
        chart_img = Image(chart_buf, width=W, height=W * 0.42)
        chart_elements.append(chart_img)
        chart_elements.append(Paragraph(
            "Dashed lines = market indices (shown only if selected in app).",
            s_small
        ))
    chart_elements.append(Spacer(1, 8))
    chart_elements.append(HRFlowable(
        width=W, thickness=0.3,
        color=colors.HexColor("#cccccc"), spaceAfter=4
    ))
    story.append(KeepTogether(chart_elements))

    # ── Comps ─────────────────────────────────────────────────
    if comps_data and subject_fund:
        all_tickers = [report['ticker']] + list(comps_data.keys())
        all_data    = {report['ticker']: subject_fund, **comps_data}

        metric_keys = [
            ("Trailing P/E",   "1_trailing_pe"),
            ("Forward P/E",    "2_forward_pe"),
            ("EV/EBITDA",      "3_ev_ebitda"),
            ("Price/Book",     "4_price_to_book"),
            ("Price/FCF",      "5_price_to_fcf"),
            ("Oper. Margin",   "6_operating_margin"),
            ("ROE",            "7_return_on_equity"),
            ("Rev. Growth",    "8_revenue_growth"),
            ("Debt/Equity",    "9_debt_to_equity"),
            ("FCF Yield",      "10_fcf_yield"),
        ]

        comp_header = ["Metric"] + [
            f"*{t}" if t == report['ticker'] else t
            for t in all_tickers
        ]
        comp_rows = []
        for label, key in metric_keys:
            row = [label]
            for t in all_tickers:
                d = all_data.get(t)
                row.append(d.get(key, "N/A") if d else "N/A")
            comp_rows.append(row)

        comp_data_full = [comp_header] + comp_rows
        n_comps        = len(all_tickers)
        metric_col     = 1.1 * inch
        ticker_col     = (W - metric_col) / n_comps

        style_cmds = [
            ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#111111")),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0), 7.5),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("ALIGN",         (0, 1), (0, -1),  "LEFT"),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 1), (-1, -1), 7.5),
            ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
            ("BACKGROUND",    (1, 1), (1, -1), colors.HexColor("#f0f0f0")),
            ("FONTNAME",      (1, 1), (1, -1), "Helvetica-Bold"),
        ]
        for i in range(2, len(comp_data_full), 2):
            style_cmds.append(
                ("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f9f9f9"))
            )

        comp_tbl = Table(
            comp_data_full,
            colWidths=[metric_col] + [ticker_col] * n_comps
        )
        comp_tbl.setStyle(TableStyle(style_cmds))

        story.append(KeepTogether([
            Paragraph("COMPARABLE COMPANIES ANALYSIS", s_section),
            comp_tbl,
            Paragraph(f"* = subject company ({report['ticker']})", s_small),
            Spacer(1, 8),
            HRFlowable(width=W, thickness=0.3,
                       color=colors.HexColor("#cccccc"), spaceAfter=4),
        ]))

    # ── Thesis ────────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph("INVESTMENT THESIS", s_section),
        Paragraph(report['investment_thesis'], s_body),
        Spacer(1, 8),
        HRFlowable(width=W, thickness=0.3,
                   color=colors.HexColor("#cccccc"), spaceAfter=4),
    ]))

    # ── Risks ─────────────────────────────────────────────────
    risks_elements = [Paragraph("KEY RISKS", s_section)]
    for i, risk in enumerate(report['key_risks'], 1):
        risks_elements.append(Paragraph(f"<b>{i}.</b> {risk}", s_body))
    risks_elements += [
        Spacer(1, 10),
        HRFlowable(width=W, thickness=0.3,
                   color=colors.HexColor("#cccccc"), spaceAfter=4),
    ]
    story.append(KeepTogether(risks_elements))

    # ── Disclaimer ────────────────────────────────────────────
    story.append(Paragraph(
        "DISCLAIMER: This report is AI-generated for informational purposes only. "
        "It does not constitute investment advice. All data sourced from Yahoo Finance "
        "with a 15-minute delay. Verify all figures independently before making any "
        "investment decisions.",
        s_small
    ))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ─── AGENT ───────────────────────────────────────────────────

data_tools = [
    {
        "name": "get_stock_price",
        "description": "Fetch the most recent closing price for a stock ticker.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"]
        }
    },
    {
        "name": "get_historical_return",
        "description": "Compute total percentage return of a stock over N years.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "years":  {"type": "integer"}
            },
            "required": ["ticker", "years"]
        }
    },
    {
        "name": "get_fundamentals",
        "description": (
            "Fetch top 10 value investor fundamentals: trailing P/E, "
            "forward P/E, EV/EBITDA, price/book, price/FCF, operating margin, "
            "ROE, revenue growth, debt/equity, FCF yield."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"]
        }
    }
]


def execute_tool(name, inputs):
    if name == "get_stock_price":
        return get_stock_price(**inputs)
    elif name == "get_historical_return":
        return get_historical_return(**inputs)
    elif name == "get_fundamentals":
        return get_fundamentals(**inputs)
    return f"Error: unknown tool '{name}'"


def gather_market_data(ticker, status):
    system_prompt = """You are a quantitative equity research analyst.
    Gather ALL of the following for the given ticker — do not skip any:
      1. Current price
      2. 1-year historical return
      3. 5-year historical return
      4. Fundamental data (top 10 value investor metrics)"""

    messages = [{
        "role": "user",
        "content": (
            f"Gather all market data for {ticker}. "
            f"Fetch current price, 1-year return, 5-year return, and fundamentals."
        )
    }]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2048,
            system=system_prompt,
            tools=data_tools,
            messages=messages
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            final_text = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            return messages, final_text

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                status.update(label=f"Fetching: {block.name}({block.input})...")
                result     = execute_tool(block.name, block.input)
                result_str = (
                    json.dumps(result) if isinstance(result, dict)
                    else str(result)
                )
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result_str
                })
        messages.append({"role": "user", "content": tool_results})


research_report_tool = {
    "name": "produce_research_report",
    "description": "Produce a structured equity research report from gathered data.",
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker":              {"type": "string"},
            "company_name":        {"type": "string"},
            "sector":              {"type": "string"},
            "industry":            {"type": "string"},
            "current_price":       {"type": "number"},
            "fifty_two_week_high": {"type": "string"},
            "fifty_two_week_low":  {"type": "string"},
            "one_year_return_pct": {"type": "number"},
            "five_year_return_pct":{"type": "number"},
            "market_cap":          {"type": "string"},
            "trailing_pe":         {"type": "string"},
            "forward_pe":          {"type": "string"},
            "ev_ebitda":           {"type": "string"},
            "price_to_book":       {"type": "string"},
            "price_to_fcf":        {"type": "string"},
            "operating_margin":    {"type": "string"},
            "return_on_equity":    {"type": "string"},
            "revenue_growth_yoy":  {"type": "string"},
            "debt_to_equity":      {"type": "string"},
            "fcf_yield":           {"type": "string"},
            "ai_rating": {
                "type": "string",
                "enum": ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
            },
            "ai_rating_rationale": {"type": "string"},
            "investment_thesis":   {"type": "string"},
            "key_risks": {
                "type":  "array",
                "items": {"type": "string"}
            },
            "one_line_summary": {"type": "string"}
        },
        "required": [
            "ticker", "company_name", "sector", "industry",
            "current_price", "fifty_two_week_high", "fifty_two_week_low",
            "one_year_return_pct", "five_year_return_pct", "market_cap",
            "trailing_pe", "forward_pe", "ev_ebitda",
            "price_to_book", "price_to_fcf", "operating_margin",
            "return_on_equity", "revenue_growth_yoy", "debt_to_equity",
            "fcf_yield", "ai_rating", "ai_rating_rationale",
            "investment_thesis", "key_risks", "one_line_summary"
        ]
    }
}


def generate_research_report(ticker, conversation_history, data_summary, status):
    status.update(label="Generating structured report...")

    system_prompt = """You are a senior equity research analyst.
    Rules:
    - Use ONLY actual numbers from gathered data. No estimates.
    - AI rating must cite specific metrics.
    - Thesis: 2-3 sentences, specific, data-grounded.
    - Exactly 3 risks, one crisp sentence each.
    - One-line summary under 20 words."""

    messages = conversation_history + [{
        "role": "user",
        "content": (
            f"Produce a structured research report for {ticker}.\n\n"
            f"Data summary: {data_summary}"
        )
    }]

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        system=system_prompt,
        tools=[research_report_tool],
        tool_choice={"type": "tool", "name": "produce_research_report"},
        messages=messages
    )

    tool_use = next(b for b in response.content if b.type == "tool_use")
    return tool_use.input


# ─── TEXT FORMATTER ──────────────────────────────────────────

def build_report_text(r, subject_fundamentals=None, comps_data=None):
    W    = 60
    div  = "=" * W
    thin = "-" * W

    def row(label, value, width=24):
        val = value if value else "N/A"
        return "" if val == "N/A" else f"\n  {label:<{width}}{val}"

    output = f"""
{div}
  EQUITY RESEARCH — AI GENERATED
{div}
  {r['company_name']} ({r['ticker']})
  {r['sector']} | {r['industry']}
  Generated: {datetime.now().strftime('%B %d, %Y')}
{thin}
  AI RATING: {r['ai_rating'].upper()}
  {r['one_line_summary']}

  Rationale: {r['ai_rating_rationale']}
{div}
  PRICE & PERFORMANCE
{thin}{row('Current Price:', f"${r['current_price']:.2f}")}{row('52-Week Range:', f"{r['fifty_two_week_low']} – {r['fifty_two_week_high']}")}{row('1-Year Return:', f"{r['one_year_return_pct']:.1f}%")}{row('5-Year Return:', f"{r['five_year_return_pct']:.1f}%")}{row('Market Cap:', r['market_cap'])}
{div}
  TOP 10 VALUE INVESTOR FUNDAMENTALS
{thin}
  VALUATION{row('1. Trailing P/E:', r['trailing_pe'])}{row('2. Forward P/E:', r['forward_pe'])}{row('3. EV / EBITDA:', r['ev_ebitda'])}{row('4. Price / Book:', r['price_to_book'])}{row('5. Price / FCF:', r['price_to_fcf'])}

  QUALITY & GROWTH{row('6. Operating Margin:', r['operating_margin'])}{row('7. Return on Equity:', r['return_on_equity'])}{row('8. Revenue Growth:', r['revenue_growth_yoy'])}{row('9. Debt / Equity:', r['debt_to_equity'])}{row('10. FCF Yield:', r['fcf_yield'])}"""

    if subject_fundamentals and comps_data:
        LABEL_W = 20
        COL_W   = 11
        all_tickers = [r['ticker']] + list(comps_data.keys())
        all_data    = {r['ticker']: subject_fundamentals, **comps_data}
        total_w     = LABEL_W + (COL_W * len(all_tickers))
        thin2       = "  " + "-" * total_w

        METRICS = [
            ("VALUATION",        None),
            ("1. Trailing P/E",  "1_trailing_pe"),
            ("2. Forward P/E",   "2_forward_pe"),
            ("3. EV / EBITDA",   "3_ev_ebitda"),
            ("4. Price / Book",  "4_price_to_book"),
            ("5. Price / FCF",   "5_price_to_fcf"),
            ("",                 None),
            ("QUALITY & GROWTH", None),
            ("6. Oper. Margin",  "6_operating_margin"),
            ("7. ROE",           "7_return_on_equity"),
            ("8. Rev. Growth",   "8_revenue_growth"),
            ("9. Debt / Equity", "9_debt_to_equity"),
            ("10. FCF Yield",    "10_fcf_yield"),
        ]

        header = f"  {'METRIC':<{LABEL_W}}"
        names  = f"  {'':<{LABEL_W}}"
        for t in all_tickers:
            label = f"[{t}]" if t == r['ticker'] else t
            name  = all_data[t].get("company_name", t)[:10] \
                    if all_data.get(t) else t
            header += f"{label:>{COL_W}}"
            names  += f"{name:>{COL_W}}"

        comp_block = f"\n{thin2}\n{header}\n{names}\n{thin2}"
        for label, key in METRICS:
            if key is None:
                comp_block += f"\n  {label}" if label else "\n"
                continue
            row_str = f"\n  {label:<{LABEL_W}}"
            for t in all_tickers:
                d   = all_data.get(t)
                val = d.get(key, "N/A") if d else "N/A"
                row_str += f"{val:>{COL_W}}"
            comp_block += row_str
        comp_block += f"\n{thin2}"

        output += f"\n{div}\n  COMPARABLE COMPANIES ANALYSIS"
        output += comp_block

    output += f"""
{div}
  INVESTMENT THESIS
{thin}
  {r['investment_thesis']}
{div}
  KEY RISKS
{thin}"""

    for i, risk in enumerate(r['key_risks'], 1):
        output += f"\n  {i}. {risk}"

    output += f"""
{div}
  DISCLAIMER: AI-generated. Not investment advice.
  Verify all data independently before making decisions.
{div}"""

    return output


# ─── RENDER FUNCTION ─────────────────────────────────────────

def render_report(report, subject_fund, comps_data, comp_tickers,
                  chart_period, normalize, selected_indices=None):

    period_code = PERIOD_MAP[chart_period]

    st.divider()

    rating_colors = {
        "Strong Buy":  "🟢",
        "Buy":         "🟩",
        "Hold":        "🟡",
        "Sell":        "🟠",
        "Strong Sell": "🔴"
    }
    emoji = rating_colors.get(report['ai_rating'], "⚪")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader(f"{report['company_name']} ({report['ticker']})")
        st.caption(f"{report['sector']} | {report['industry']}")
    with col2:
        st.metric("AI Rating", f"{emoji} {report['ai_rating']}")
    with col3:
        st.metric("Current Price", f"${report['current_price']:.2f}")

    st.info(f"**{report['one_line_summary']}**")
    st.caption(f"Rating rationale: {report['ai_rating_rationale']}")

    st.divider()

    st.subheader("Price & Performance")
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Current Price",  f"${report['current_price']:.2f}")
    p2.metric("1-Year Return",  f"{report['one_year_return_pct']:.1f}%")
    p3.metric("5-Year Return",  f"{report['five_year_return_pct']:.1f}%")
    p4.metric("Market Cap",     report['market_cap'])
    p5.metric("52-Wk Range",
              f"{report['fifty_two_week_low']} – {report['fifty_two_week_high']}")

    st.divider()

    st.subheader("📊 Price History")
    with st.spinner("Loading chart..."):
        fig = build_price_chart(
            subject_ticker=report['ticker'],
            comp_tickers=comp_tickers,
            period=period_code,
            normalize=normalize,
            selected_indices=selected_indices
        )

    if fig:
        if not normalize:
            st.caption(
                "💡 Tip: Index prices are much higher than most stocks. "
                "Toggle **Normalize to 100** in the sidebar for "
                "apples-to-apples comparison."
            )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Click legend items to show/hide series. "
            "Indices shown as dotted lines. "
            "Double-click to isolate a series."
        )
    else:
        st.warning("Could not load chart data.")

    st.divider()

    # ── Fundamentals tables (st.dataframe fixes the "10." issue) ──
    st.subheader("Top 10 Value Investor Fundamentals")
    fund_c1, fund_c2 = st.columns(2)

    with fund_c1:
        st.markdown("**Valuation**")
        st.dataframe(
            pd.DataFrame({
                "Metric": [
                    "1. Trailing P/E", "2. Forward P/E",
                    "3. EV / EBITDA",  "4. Price / Book",
                    "5. Price / FCF"
                ],
                report['ticker']: [
                    report['trailing_pe'], report['forward_pe'],
                    report['ev_ebitda'],   report['price_to_book'],
                    report['price_to_fcf']
                ]
            }),
            hide_index=True,
            use_container_width=True
        )

    with fund_c2:
        st.markdown("**Quality & Growth**")
        st.dataframe(
            pd.DataFrame({
                "Metric": [
                    "6. Operating Margin", "7. Return on Equity",
                    "8. Revenue Growth",   "9. Debt / Equity",
                    "10. FCF Yield"
                ],
                report['ticker']: [
                    report['operating_margin'], report['return_on_equity'],
                    report['revenue_growth_yoy'], report['debt_to_equity'],
                    report['fcf_yield']
                ]
            }),
            hide_index=True,
            use_container_width=True
        )

    # ── Comps tables ──────────────────────────────────────────
    if comps_data:
        st.divider()
        st.subheader("Comparable Companies Analysis")

        all_tickers = [report['ticker']] + list(comps_data.keys())
        all_data    = {report['ticker']: subject_fund, **comps_data}

        metrics_val = [
            ("1. Trailing P/E",  "1_trailing_pe"),
            ("2. Forward P/E",   "2_forward_pe"),
            ("3. EV / EBITDA",   "3_ev_ebitda"),
            ("4. Price / Book",  "4_price_to_book"),
            ("5. Price / FCF",   "5_price_to_fcf"),
        ]
        metrics_qual = [
            ("6. Oper. Margin",  "6_operating_margin"),
            ("7. ROE",           "7_return_on_equity"),
            ("8. Rev. Growth",   "8_revenue_growth"),
            ("9. Debt / Equity", "9_debt_to_equity"),
            ("10. FCF Yield",    "10_fcf_yield"),
        ]

        def build_comp_df(metrics):
            data = {"Metric": [m[0] for m in metrics]}
            for t in all_tickers:
                label = f"★ {t}" if t == report['ticker'] else t
                d     = all_data.get(t)
                data[label] = [
                    d.get(key, "N/A") if d else "N/A"
                    for _, key in metrics
                ]
            return pd.DataFrame(data)

        comp_c1, comp_c2 = st.columns(2)
        with comp_c1:
            st.markdown("**Valuation**")
            st.dataframe(
                build_comp_df(metrics_val),
                hide_index=True,
                use_container_width=True
            )
        with comp_c2:
            st.markdown("**Quality & Growth**")
            st.dataframe(
                build_comp_df(metrics_qual),
                hide_index=True,
                use_container_width=True
            )

        st.caption(f"★ = subject company ({report['ticker']})")

    st.divider()

    thesis_col, risks_col = st.columns(2)
    with thesis_col:
        st.subheader("Investment Thesis")
        st.write(report['investment_thesis'])
    with risks_col:
        st.subheader("Key Risks")
        for i, risk in enumerate(report['key_risks'], 1):
            st.warning(f"**{i}.** {risk}")

    st.divider()

    st.subheader("Download Report")
    date_str      = datetime.now().strftime('%Y%m%d')
    filename_stem = f"{report['ticker']}_research_{date_str}"

    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        report_text = build_report_text(report, subject_fund, comps_data)
        st.download_button(
            label="⬇️ Download as .txt",
            data=report_text,
            file_name=f"{filename_stem}.txt",
            mime="text/plain",
            use_container_width=True
        )

    with dl_col2:
        with st.spinner("Preparing PDF..."):
            pdf_bytes = generate_pdf(
                report, subject_fund, comps_data,
                comp_tickers, chart_period, normalize,
                selected_indices
            )
        st.download_button(
            label="⬇️ Download as PDF",
            data=pdf_bytes,
            file_name=f"{filename_stem}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    st.caption(
        "DISCLAIMER: AI-generated report for educational purposes only. "
        "Not investment advice. Verify all data before making decisions."
    )


# ─── STREAMLIT UI ────────────────────────────────────────────

st.set_page_config(
    page_title="Equity Research Assistant",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Equity Research Assistant")
st.caption("AI-generated research reports — not investment advice")

with st.sidebar:
    st.header("Report Parameters")

    ticker_input = st.text_input(
        "Stock Ticker",
        placeholder="e.g. AVGO",
        help="Enter the ticker symbol of the company to research"
    ).upper().strip()

    comps_input = st.text_input(
        "Comparable Companies (optional)",
        placeholder="e.g. MRVL, QCOM, CSCO, INTC",
        help="Comma-separated list of comp tickers"
    )

    st.divider()
    st.subheader("Chart Settings")

    chart_period = st.selectbox(
        "Time Period",
        options=list(PERIOD_MAP.keys()),
        index=1
    )

    normalize = st.toggle(
        "Normalize to 100",
        value=False,
        help="Rebase all series to 100 at start date for % comparison"
    )

    st.markdown("**Show indices on chart:**")
    show_sp500  = st.checkbox("S&P 500",   value=False)
    show_nasdaq = st.checkbox("NASDAQ",    value=False)
    show_dow    = st.checkbox("Dow Jones", value=False)

    st.divider()

    generate_btn = st.button(
        "Generate Report",
        type="primary",
        disabled=not ticker_input,
        use_container_width=True
    )

    st.divider()
    st.caption("Built with Claude + yfinance + Plotly")
    st.caption("Yahoo Finance data — 15-min delay")


selected_indices = []
if show_sp500:
    selected_indices.append("S&P 500")
if show_nasdaq:
    selected_indices.append("NASDAQ")
if show_dow:
    selected_indices.append("Dow Jones")


if generate_btn and ticker_input:

    with st.spinner(f"Validating {ticker_input}..."):
        is_valid, company_name = validate_ticker(ticker_input)

    if not is_valid:
        st.error(
            f"❌ **{ticker_input}** doesn't appear to be a valid ticker. "
            f"Please check the symbol and try again."
        )
        st.stop()

    comps_data   = {}
    comp_tickers = []
    if comps_input.strip():
        raw_comps = [
            c.upper().strip()
            for c in comps_input.split(",") if c.strip()
        ]
        comps_data, invalid_tickers = get_comps_data(raw_comps)
        if invalid_tickers:
            st.warning(
                f"⚠️ Skipped invalid ticker(s): "
                f"**{', '.join(invalid_tickers)}**. "
                f"Please verify the symbols."
            )
        comp_tickers = list(comps_data.keys())

    with st.status(
        f"Researching {ticker_input}...", expanded=True
    ) as status:

        status.update(label=f"Gathering market data for {ticker_input}...")
        history, summary = gather_market_data(ticker_input, status)

        status.update(label="Fetching subject fundamentals...")
        subject_fund = get_fundamentals(ticker_input)
        if not isinstance(subject_fund, dict):
            subject_fund = None

        report = generate_research_report(
            ticker_input, history, summary, status
        )

        st.session_state.report       = report
        st.session_state.subject_fund = subject_fund
        st.session_state.comps_data   = comps_data
        st.session_state.comp_tickers = comp_tickers

        status.update(label="Report ready!", state="complete")

    render_report(
        report, subject_fund, comps_data, comp_tickers,
        chart_period, normalize, selected_indices
    )

elif "report" in st.session_state and not generate_btn:
    render_report(
        st.session_state.report,
        st.session_state.subject_fund,
        st.session_state.comps_data,
        st.session_state.comp_tickers,
        chart_period,
        normalize,
        selected_indices
    )

else:
    st.info("👈 Enter a ticker in the sidebar and click Generate Report.")