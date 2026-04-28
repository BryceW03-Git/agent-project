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
from datetime import datetime, timedelta
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

ACTION_MAP = {
    "Reit":    "Reiterated",
    "Main":    "Maintained",
    "Up":      "Upgraded",
    "Down":    "Downgraded",
    "Init":    "Initiated",
    "Assumed": "Assumed",
}

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

        def calc_roic(stock):
            try:
                bs  = stock.balance_sheet
                fin = stock.financials
                if bs is None or fin is None or bs.empty or fin.empty:
                    return "N/A"

                op_income = (fin.loc["Operating Income"].iloc[0]
                             if "Operating Income" in fin.index else None)

                tax_provision = (fin.loc["Tax Provision"].iloc[0]
                                 if "Tax Provision" in fin.index else None)

                if not op_income:
                    return "N/A"

                # NOPAT = Operating Income - taxes paid (floor tax benefit at zero)
                # Negative tax provision = government tax benefit = treat as zero tax paid
                tax_paid = max(0, tax_provision) if tax_provision is not None else op_income * 0.21
                nopat = op_income - tax_paid

                # Invested Capital = Equity + Total Debt - Cash
                equity = (bs.loc["Stockholders Equity"].iloc[0]
                          if "Stockholders Equity" in bs.index else None)
                debt   = (bs.loc["Long Term Debt"].iloc[0]
                          if "Long Term Debt" in bs.index else 0)
                cash   = (bs.loc["Cash And Cash Equivalents"].iloc[0]
                          if "Cash And Cash Equivalents" in bs.index else 0)

                if equity:
                    invested_capital = equity + (debt or 0) - (cash or 0)
                    if invested_capital > 0:
                        roic = (nopat / invested_capital) * 100
                        return f"{roic:.1f}%"
                return "N/A"
            except Exception:
                return "N/A"

        return {
            "company_name":        info.get("longName", ticker),
            "sector":              info.get("sector", "N/A"),
            "industry":            info.get("industry", "N/A"),
            "market_cap":          dollar_b(market_cap),
            "fifty_two_week_high": f"${info.get('fiftyTwoWeekHigh', 0):.2f}",
            "fifty_two_week_low":  f"${info.get('fiftyTwoWeekLow', 0):.2f}",
            "1_trailing_pe":       multiple(info.get("trailingPE")),
            "2_forward_pe":        multiple(info.get("forwardPE")),
            "3_ev_ebitda":         multiple(ev_ebitda),
            "4_price_to_book":     multiple(info.get("priceToBook")),
            "5_price_to_fcf":      multiple(price_to_fcf),
            "6_operating_margin":  pct(info.get("operatingMargins")),
            "7_roic":              calc_roic(stock),
            "8_return_on_equity":  pct(info.get("returnOnEquity")),
            "9_revenue_growth":    pct(info.get("revenueGrowth")),
            "10_debt_to_equity":   multiple(info.get("debtToEquity"), decimals=2),
            "11_fcf_yield":        pct(fcf_yield),
        }
    except Exception:
        return None


def validate_ticker(ticker):
    import time
    for attempt in range(2):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if info and len(info) > 5:
                return True, info.get("longName", ticker)
            if attempt == 0:
                time.sleep(2)
        except Exception:
            if attempt == 0:
                time.sleep(2)
    return True, ticker  # Allow through with warning rather than blocking


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


# ─── ANALYST DATA ────────────────────────────────────────────

def get_analyst_data(ticker):
    try:
        stock = yf.Ticker(ticker)

        # Price targets
        targets = {}
        try:
            pt = stock.analyst_price_targets
            if pt and isinstance(pt, dict):
                targets = {
                    "mean":   pt.get("mean"),
                    "high":   pt.get("high"),
                    "low":    pt.get("low"),
                    "median": pt.get("median"),
                }
        except Exception:
            pass

        # Upgrade / downgrade history — last 3 months
        recs_df = None
        cutoff  = datetime.now() - timedelta(days=90)

        try:
            raw = stock.upgrades_downgrades
            if raw is not None and not raw.empty:
                raw = raw.copy()
                if hasattr(raw.index, "tz") and raw.index.tz is not None:
                    raw.index = raw.index.tz_convert(None)

                raw = raw[raw.index >= cutoff]

                if not raw.empty:
                    raw = raw.reset_index()
                    date_col   = next((c for c in raw.columns
                                       if "date" in c.lower()), None)
                    firm_col   = next((c for c in raw.columns
                                       if "firm" in c.lower()), None)
                    to_col     = next((c for c in raw.columns
                                       if "tograde" in c.lower()
                                       or "to grade" in c.lower()), None)
                    from_col   = next((c for c in raw.columns
                                       if "fromgrade" in c.lower()
                                       or "from grade" in c.lower()), None)
                    action_col = next((c for c in raw.columns
                                       if "action" in c.lower()), None)

                    if date_col and firm_col and to_col:
                        raw_actions = (
                            raw[action_col].str.capitalize()
                            if action_col
                            else pd.Series([""] * len(raw))
                        )
                        recs_df = pd.DataFrame({
                            "Date":      pd.to_datetime(raw[date_col]).dt.date,
                            "Firm":      raw[firm_col].str.strip(),
                            "Rating":    raw[to_col].str.strip(),
                            "FromGrade": (raw[from_col].str.strip()
                                          if from_col else ""),
                            "Action":    raw_actions.map(
                                          lambda x: ACTION_MAP.get(x, x)
                                         ),
                        })
                        recs_df = recs_df.sort_values(
                            "Date", ascending=False
                        ).reset_index(drop=True)

        except Exception:
            pass

        return targets, recs_df

    except Exception:
        return {}, None


def get_financial_history(ticker, years=4):
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        if financials is None or financials.empty:
            return f"No financial history available for {ticker}"

        years = min(int(years), 10)

        result = {}
        rows_wanted = ["Total Revenue", "Net Income",
                       "Operating Income", "Gross Profit"]
        for row in rows_wanted:
            if row in financials.index:
                row_data = financials.loc[row].iloc[:years]
                result[row] = {
                    str(col.year): f"${val/1e9:.2f}B"
                    for col, val in row_data.items()
                    if val is not None and not pd.isna(val)
                }

        info = stock.info
        margins = {}
        if info.get("grossMargins"):
            margins["Gross Margin"] = f"{info['grossMargins']*100:.1f}%"
        if info.get("operatingMargins"):
            margins["Operating Margin"] = f"{info['operatingMargins']*100:.1f}%"
        if info.get("profitMargins"):
            margins["Net Margin"] = f"{info['profitMargins']*100:.1f}%"

        result["Current Margins"] = margins
        result["Years Shown"] = years
        return result
    except Exception as e:
        return f"Error fetching financial history for {ticker}: {str(e)}"


def get_recent_news(ticker, max_articles=5):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return f"No recent news found for {ticker}"

        articles = []
        for item in news[:max_articles]:
            article = {
                "title":     item.get("title", "No title"),
                "publisher": item.get("publisher", "Unknown"),
                "summary":   item.get("summary", "No summary available")[:300],
                "published": (
                    datetime.fromtimestamp(
                        item.get("providerPublishTime", 0)
                    ).strftime("%Y-%m-%d")
                    if item.get("providerPublishTime") else "Unknown date"
                ),
            }
            articles.append(article)

        return {
            "ticker":        ticker,
            "article_count": len(articles),
            "articles":      articles,
        }
    except Exception as e:
        return f"Error fetching news for {ticker}: {str(e)}"


def get_unique_firms(recs_df):
    if recs_df is None or recs_df.empty:
        return []
    most_recent = recs_df.drop_duplicates(subset=["Firm"], keep="first")
    return sorted(most_recent["Firm"].tolist())


# ─── PLOTLY CHART ────────────────────────────────────────────

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
            orientation="h", yanchor="bottom",
            y=1.02, xanchor="right", x=1
        ),
        height=500,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    )
    return fig


# ─── MATPLOTLIB CHART (PDF) ──────────────────────────────────

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

        ax.plot(x, series.values, label=name,
                color=CHART_COLORS[i % len(CHART_COLORS)],
                linewidth=2.5 if is_subject else 1.2,
                linestyle="--" if is_index else "-",
                zorder=3 if is_subject else 2)

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
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9,
              edgecolor="#cccccc", ncol=min(len(all_series), 4))

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
                 selected_indices=None,
                 analyst_targets=None, analyst_recs=None,
                 selected_firms=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.2 * inch, rightMargin=0.2 * inch,
        topMargin=0.3 * inch,  bottomMargin=0.2 * inch,
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

    def hr(thickness=0.3, color="#cccccc", space_after=4):
        """Horizontal rule as a full-width Table so it always spans exactly W."""
        tbl = Table([['']], colWidths=[W], rowHeights=[thickness])
        tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (0,0), colors.HexColor(color)),
            ('TOPPADDING',    (0,0), (0,0), 0),
            ('BOTTOMPADDING', (0,0), (0,0), 0),
            ('LEFTPADDING',   (0,0), (0,0), 0),
            ('RIGHTPADDING',  (0,0), (0,0), 0),
        ]))
        tbl.spaceAfter = space_after
        return tbl

    story = []

    # ── Header ────────────────────────────────────────────────
    rating_bg = {
        "Strong Buy":  "#1a7f3c", "Buy": "#2e9e57",
        "Hold": "#b8860b", "Sell": "#c0392b",
        "Strong Sell": "#922b21",
    }.get(report['ai_rating'], "#444444")

    # Compute analyst consensus counts for the header badge
    _pdf_show_cons = False
    _pdf_n_bull = _pdf_n_neut = _pdf_n_bear = 0
    if analyst_recs is not None and not analyst_recs.empty:
        _pdf_cf = analyst_recs.copy()
        if selected_firms is not None:
            _pdf_cf = _pdf_cf[_pdf_cf["Firm"].isin(selected_firms)]
        if not _pdf_cf.empty:
            _bull_kws = {"buy", "outperform", "overweight", "strong buy"}
            _neut_kws = {"hold", "neutral", "equal weight", "equal-weight",
                         "market perform", "sector perform", "in-line", "sector weight"}
            _bear_kws = {"sell", "underperform", "underweight", "strong sell"}
            _rv = _pdf_cf["Rating"].str.lower().str.strip()
            _pdf_n_bull = int(_rv.apply(lambda r: any(k in r for k in _bull_kws)).sum())
            _pdf_n_neut = int(_rv.apply(lambda r: any(k in r for k in _neut_kws)).sum())
            _pdf_n_bear = int(_rv.apply(lambda r: any(k in r for k in _bear_kws)).sum())
            _pdf_show_cons = True

    if _pdf_show_cons:
        cons_str = (f"{_pdf_n_bull} Bullish  |  "
                    f"{_pdf_n_neut} Neutral  |  "
                    f"{_pdf_n_bear} Bearish")
        rating_tbl = Table(
            [[Paragraph(f"AI Rating: {report['ai_rating']}", s_rating),
              Paragraph(f"ANALYST CONSENSUS:   {cons_str}", s_rating)]],
            colWidths=[2.5 * inch, W - 2.5 * inch]
        )
        rating_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, 0), colors.HexColor(rating_bg)),
            ("BACKGROUND",    (1, 0), (1, 0), colors.HexColor("#2c3e50")),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (1, 0), (1, 0), 14),
        ]))
    else:
        rating_tbl = Table(
            [[Paragraph(f"AI Rating: {report['ai_rating']}", s_rating)]],
            colWidths=[2.5 * inch]
        )
        rating_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor(rating_bg)),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))

    story.append(KeepTogether([
        Paragraph(f"{report['company_name']} ({report['ticker']})", s_title),
        Paragraph(f"{report['sector']} &nbsp;|&nbsp; {report['industry']}",
                  s_subtitle),
        Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y')} &nbsp;|&nbsp; "
            f"AI-Generated Equity Research — Not Investment Advice", s_date),
        hr(thickness=1.5, color="#111111", space_after=8),
        rating_tbl,
        Spacer(1, 6),
        Paragraph(report['one_line_summary'], s_oneliner),
        Paragraph(f"<i>Rationale: {report['ai_rating_rationale']}</i>",
                  s_rationale),
        hr(space_after=6),
    ]))

    # ── Performance ───────────────────────────────────────────
    perf_headers = ["Current Price", "1-Year Return", "5-Year Return",
                    "Market Cap", "52-Week Range"]
    perf_values  = [
        f"${report['current_price']:.2f}",
        f"{report['one_year_return_pct']:.1f}%",
        f"{report['five_year_return_pct']:.1f}%",
        report.get('market_cap', 'N/A'),
        f"{report.get('fifty_two_week_low', 'N/A')} – {report.get('fifty_two_week_high', 'N/A')}"
    ]
    col_w    = W / len(perf_headers)
    perf_tbl = Table([perf_headers, perf_values],
                     colWidths=[col_w] * len(perf_headers))
    perf_tbl.setStyle(perf_table_style())

    story.append(KeepTogether([
        Paragraph("PRICE &amp; PERFORMANCE", s_section),
        perf_tbl, Spacer(1, 8),
        hr(),
    ]))

    # ── Fundamentals ──────────────────────────────────────────
    val_rows = [
        ["1. Trailing P/E",   report.get('trailing_pe', 'N/A')],
        ["2. Forward P/E",    report.get('forward_pe', 'N/A')],
        ["3. EV / EBITDA",    report.get('ev_ebitda', 'N/A')],
        ["4. Price / Book",   report.get('price_to_book', 'N/A')],
        ["5. Price / FCF",    report.get('price_to_fcf', 'N/A')],
    ]
    qual_rows = [
        ["6. Operating Margin", report.get('operating_margin', 'N/A')],
        ["7. ROIC",             report.get('roic', 'N/A')],
        ["8. Return on Equity", report.get('return_on_equity', 'N/A')],
        ["9. Revenue Growth",   report.get('revenue_growth_yoy', 'N/A')],
        ["10. Debt / Equity",   report.get('debt_to_equity', 'N/A')],
        ["11. FCF Yield",       report.get('fcf_yield', 'N/A')],
    ]

    def make_fund_table(rows, col_widths):
        data = [["Metric", report['ticker']]] + rows
        tbl  = Table(data, colWidths=col_widths)
        tbl.setStyle(header_table_style(len(data)))
        return tbl

    half     = W / 2
    gap      = 6  # points between the two side-by-side tables
    val_tbl  = make_fund_table(val_rows,  [(half - gap) * 0.65, (half - gap) * 0.35])
    qual_tbl = make_fund_table(qual_rows, [half * 0.65, half * 0.35])
    fund_layout = Table([[val_tbl, qual_tbl]],
                        colWidths=[half, half])
    fund_layout.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (0, -1),  gap),
    ]))

    story.append(KeepTogether([
        Paragraph("TOP 10 VALUE INVESTOR FUNDAMENTALS", s_section),
        fund_layout,
        Spacer(1, 4),
        Paragraph(
            "Valuation multiples based on trailing 12-month (TTM) financials "
            "unless noted. Forward P/E based on analyst consensus estimates. "
            "Balance sheet metrics reflect most recent quarter. Revenue growth "
            "is year-over-year. ROIC = NOPAT / (Equity + Debt - Cash), "
            "where NOPAT = Operating Income minus taxes paid "
            "(tax benefits treated as zero). "
            "Source: Yahoo Finance.",
            s_small
        ),
        Spacer(1, 4),
        hr(),
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
            s_small))
    chart_elements += [Spacer(1, 8),
                       hr()]
    story.append(KeepTogether(chart_elements))

    # ── Comps ─────────────────────────────────────────────────
    if comps_data and subject_fund:
        all_tickers = [report['ticker']] + list(comps_data.keys())
        all_data    = {report['ticker']: subject_fund, **comps_data}

        metric_keys = [
            ("Trailing P/E",  "1_trailing_pe"),
            ("Forward P/E",   "2_forward_pe"),
            ("EV/EBITDA",     "3_ev_ebitda"),
            ("Price/Book",    "4_price_to_book"),
            ("Price/FCF",     "5_price_to_fcf"),
            ("Oper. Margin",  "6_operating_margin"),
            ("ROIC",          "7_roic"),
            ("ROE",           "8_return_on_equity"),
            ("Rev. Growth",   "9_revenue_growth"),
            ("Debt/Equity",   "10_debt_to_equity"),
            ("FCF Yield",     "11_fcf_yield"),
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
        n_comps    = len(all_tickers)
        metric_col = 1.1 * inch
        ticker_col = (W - metric_col) / n_comps

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
            ("FONTNAME",      (1, 1), (1, -1), "Helvetica-Bold"),
        ]
        for i in range(2, len(comp_data_full), 2):
            style_cmds.append(
                ("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f9f9f9"))
            )
        comp_tbl = Table(comp_data_full,
                         colWidths=[metric_col] + [ticker_col] * n_comps)
        comp_tbl.setStyle(TableStyle(style_cmds))

        story.append(KeepTogether([
            Paragraph("COMPARABLE COMPANIES ANALYSIS", s_section),
            comp_tbl,
            Paragraph(f"* = subject company ({report['ticker']})", s_small),
            Spacer(1, 8),
            hr(),
        ]))

    # ── Analyst Recommendations ───────────────────────────────
    has_targets = analyst_targets and any(
        v for v in analyst_targets.values() if v
    )
    has_recs = (analyst_recs is not None and not analyst_recs.empty)

    if has_targets or has_recs:
        analyst_elements = [Paragraph("ANALYST RECOMMENDATIONS", s_section)]

        if has_targets:
            current = report['current_price']
            mean_t  = analyst_targets.get("mean")
            high_t  = analyst_targets.get("high")
            low_t   = analyst_targets.get("low")
            med_t   = analyst_targets.get("median")

            def fmt_target(val):
                return f"${val:.2f}" if val else "N/A"

            def fmt_updown(val):
                if not val or not current:
                    return ""
                pct = ((val - current) / current) * 100
                return f"({pct:+.1f}%)"

            pt_headers = ["Mean Target", "Upside/Downside",
                          "High Target", "Low Target", "Median Target"]
            pt_values  = [
                fmt_target(mean_t), fmt_updown(mean_t),
                fmt_target(high_t), fmt_target(low_t), fmt_target(med_t),
            ]
            pt_col_w = W / len(pt_headers)
            pt_tbl   = Table([pt_headers, pt_values],
                             colWidths=[pt_col_w] * len(pt_headers))
            pt_tbl.setStyle(perf_table_style())
            analyst_elements.append(pt_tbl)
            analyst_elements.append(Paragraph(
                "Price targets represent the aggregate of all price targets "
                "submitted by analysts to Yahoo Finance.",
                s_small
            ))
            analyst_elements.append(Spacer(1, 6))

        if has_recs:
            filtered = analyst_recs.copy()
            if selected_firms:
                filtered = filtered[filtered["Firm"].isin(selected_firms)]

            if not filtered.empty:
                rec_header = ["Date", "Firm", "Rating", "Action"]
                rec_rows   = []
                for _, row in filtered.iterrows():
                    rec_rows.append([
                        str(row["Date"]),
                        row["Firm"],
                        row["Rating"],
                        row["Action"],
                    ])

                rec_data   = [rec_header] + rec_rows
                _date_w    = 0.9 * inch
                _rest      = W - _date_w
                col_widths = [_date_w, _rest * 0.42, _rest * 0.33, _rest * 0.25]
                rec_tbl = Table(rec_data, colWidths=col_widths)
                rec_style = [
                    ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#111111")),
                    ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
                    ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE",     (0, 0), (-1, 0), 8),
                    ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE",     (0, 1), (-1, -1), 8),
                    ("ALIGN",        (0, 0), (-1, -1), "LEFT"),
                    ("GRID",         (0, 0), (-1, -1), 0.3,
                                     colors.HexColor("#dddddd")),
                    ("TOPPADDING",   (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
                    ("LEFTPADDING",  (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ]
                for i in range(2, len(rec_data), 2):
                    rec_style.append(
                        ("BACKGROUND", (0, i), (-1, i),
                         colors.HexColor("#f7f7f7"))
                    )
                rec_tbl.setStyle(TableStyle(rec_style))
                analyst_elements.append(rec_tbl)
                analyst_elements.append(Paragraph(
                    "Ratings shown as reported by each firm. "
                    "Last 3 months only. Source: Yahoo Finance.",
                    s_small
                ))

        analyst_elements += [
            Spacer(1, 8),
            hr(),
        ]
        story.append(KeepTogether(analyst_elements))

    # ── Thesis ────────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph("INVESTMENT THESIS", s_section),
        Paragraph(report['investment_thesis'], s_body),
        Spacer(1, 8),
        hr(),
    ]))

    # ── Risks ─────────────────────────────────────────────────
    risks_elements = [Paragraph("KEY RISKS", s_section)]
    for i, risk in enumerate(report['key_risks'], 1):
        risks_elements.append(Paragraph(f"<b>{i}.</b> {risk}", s_body))
    risks_elements += [
        Spacer(1, 10),
        hr(),
    ]
    story.append(KeepTogether(risks_elements))

    # ── Disclaimer ────────────────────────────────────────────
    story.append(Paragraph(
        "DISCLAIMER: This report is AI-generated for informational purposes "
        "only. It does not constitute investment advice. All data sourced from "
        "Yahoo Finance with a 15-minute delay. Verify all figures independently "
        "before making any investment decisions.",
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
            "Fetch key value investor fundamentals: trailing P/E, "
            "forward P/E, EV/EBITDA, price/book, price/FCF, operating margin, "
            "ROIC, ROE, revenue growth, debt/equity, FCF yield."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"]
        }
    },
    {
        "name": "get_analyst_data",
        "description": (
            "Fetch analyst price targets (mean, high, low, median) and recent "
            "upgrade/downgrade history (last 3 months) for a stock ticker."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"]
        }
    },
    {
        "name": "get_financial_history",
        "description": (
            "Fetch annual revenue, net income, operating income, and gross profit "
            "history plus current margins. Supports up to 10 years of history — "
            "default is 4 years unless user specifies more. Use when asked about "
            "revenue growth trends, earnings history, or financial performance "
            "over time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "years": {
                    "type": "integer",
                    "description": (
                        "Number of years of history to return. "
                        "Default 4, maximum 10. Pick based on what the user asks for."
                    )
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_recent_news",
        "description": (
            "Fetch recent news headlines and summaries for a stock ticker. "
            "Use when asked about recent events, earnings, announcements, "
            "or news about a company."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "max_articles": {
                    "type": "integer",
                    "description": "Number of articles to return. Default 5, maximum 10."
                }
            },
            "required": ["ticker"]
        }
    },
]


def execute_tool(name, inputs):
    if name == "get_stock_price":
        return get_stock_price(**inputs)
    elif name == "get_historical_return":
        return get_historical_return(**inputs)
    elif name == "get_fundamentals":
        return get_fundamentals(**inputs)
    elif name == "get_analyst_data":
        ticker = inputs["ticker"]
        targets, recs_df = get_analyst_data(ticker)
        data = {"ticker": ticker}
        if targets:
            data["price_targets"] = {
                k: (f"${v:.2f}" if v else "N/A")
                for k, v in targets.items()
            }
        if recs_df is not None and not recs_df.empty:
            data["recent_ratings"] = [
                {
                    "date":   str(row["Date"]),
                    "firm":   row["Firm"],
                    "rating": row["Rating"],
                    "action": row["Action"],
                }
                for _, row in recs_df.head(10).iterrows()
            ]
        if len(data) == 1:
            return f"No analyst data available for {ticker}"
        return data
    elif name == "get_financial_history":
        return get_financial_history(**inputs)
    elif name == "get_recent_news":
        return get_recent_news(**inputs)
    return f"Error: unknown tool '{name}'"


def gather_market_data(ticker, status):
    system_prompt = """You are a quantitative equity research analyst.
    Gather ALL of the following — do not skip any:
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
            "roic":                {"type": "string", "description": "Return on Invested Capital as a percentage"},
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
            "roic", "return_on_equity", "revenue_growth_yoy", "debt_to_equity",
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

def build_report_text(r, subject_fundamentals=None, comps_data=None,
                      analyst_targets=None, analyst_recs=None,
                      selected_firms=None):
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
  AI Rating: {r['ai_rating']}
  {r['one_line_summary']}

  Rationale: {r['ai_rating_rationale']}
{div}
  PRICE & PERFORMANCE
{thin}{row('Current Price:', f"${r['current_price']:.2f}")}{row('52-Week Range:', f"{r['fifty_two_week_low']} – {r['fifty_two_week_high']}")}{row('1-Year Return:', f"{r['one_year_return_pct']:.1f}%")}{row('5-Year Return:', f"{r['five_year_return_pct']:.1f}%")}{row('Market Cap:', r['market_cap'])}
{div}
  TOP 10 VALUE INVESTOR FUNDAMENTALS
{thin}
  VALUATION{row('1. Trailing P/E:', r['trailing_pe'])}{row('2. Forward P/E:', r['forward_pe'])}{row('3. EV / EBITDA:', r['ev_ebitda'])}{row('4. Price / Book:', r['price_to_book'])}{row('5. Price / FCF:', r['price_to_fcf'])}

  QUALITY & GROWTH{row('6. Operating Margin:', r['operating_margin'])}{row('7. ROIC:', r['roic'])}{row('8. Return on Equity:', r['return_on_equity'])}{row('9. Revenue Growth:', r['revenue_growth_yoy'])}{row('10. Debt / Equity:', r['debt_to_equity'])}{row('11. FCF Yield:', r['fcf_yield'])}"""

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
            ("7. ROIC",          "7_roic"),
            ("8. ROE",           "8_return_on_equity"),
            ("9. Rev. Growth",   "9_revenue_growth"),
            ("10. Debt / Equity","10_debt_to_equity"),
            ("11. FCF Yield",    "11_fcf_yield"),
        ]
        header = f"  {'METRIC':<{LABEL_W}}"
        names  = f"  {'':<{LABEL_W}}"
        for t in all_tickers:
            label = f"[{t}]" if t == r['ticker'] else t
            name  = (all_data[t].get("company_name", t)[:10]
                     if all_data.get(t) else t)
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

    if analyst_targets and any(v for v in analyst_targets.values() if v):
        current = r['current_price']
        mean_t  = analyst_targets.get("mean")
        output += f"\n{div}\n  ANALYST RECOMMENDATIONS\n{thin}"
        output += (
            "\n  Price targets represent the aggregate of all price targets "
            "submitted by analysts to Yahoo Finance.\n"
        )
        if mean_t and current:
            upside = ((mean_t - current) / current) * 100
            output += f"\n  {'Mean Price Target:':<24}${mean_t:.2f} ({upside:+.1f}%)"
        if analyst_targets.get("high"):
            output += f"\n  {'High Target:':<24}${analyst_targets['high']:.2f}"
        if analyst_targets.get("low"):
            output += f"\n  {'Low Target:':<24}${analyst_targets['low']:.2f}"
        if analyst_targets.get("median"):
            output += f"\n  {'Median Target:':<24}${analyst_targets['median']:.2f}"

    if analyst_recs is not None and not analyst_recs.empty:
        filtered = analyst_recs.copy()
        if selected_firms:
            filtered = filtered[filtered["Firm"].isin(selected_firms)]
        if not filtered.empty:
            output += f"\n\n  {'Date':<12}{'Firm':<30}{'Rating':<20}Action"
            output += f"\n  {'-'*70}"
            for _, row_data in filtered.iterrows():
                output += (f"\n  {str(row_data['Date']):<12}"
                           f"{row_data['Firm']:<30}"
                           f"{row_data['Rating']:<20}"
                           f"{row_data['Action']}")

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
                  chart_period, normalize, selected_indices=None,
                  analyst_targets=None, analyst_recs=None,
                  selected_firms=None):

    period_code = PERIOD_MAP[chart_period]
    st.divider()

    # Rating + headline
    rating_colors = {
        "Strong Buy":  "🟢", "Buy": "🟩",
        "Hold": "🟡", "Sell": "🟠", "Strong Sell": "🔴"
    }
    emoji = rating_colors.get(report['ai_rating'], "⚪")

    # Pre-compute analyst consensus so it can sit alongside AI Rating in the header
    _show_cons = False
    _n_bull = _n_neut = _n_bear = 0
    if analyst_recs is not None and not analyst_recs.empty:
        _cf = analyst_recs.copy()
        if selected_firms is not None:
            _cf = _cf[_cf["Firm"].isin(selected_firms)]
        if not _cf.empty:
            _bull_kws = {"buy", "outperform", "overweight", "strong buy"}
            _neut_kws = {"hold", "neutral", "equal weight", "equal-weight",
                         "market perform", "sector perform", "in-line", "sector weight"}
            _bear_kws = {"sell", "underperform", "underweight", "strong sell"}
            _rv = _cf["Rating"].str.lower().str.strip()
            _n_bull = int(_rv.apply(lambda r: any(k in r for k in _bull_kws)).sum())
            _n_neut = int(_rv.apply(lambda r: any(k in r for k in _neut_kws)).sum())
            _n_bear = int(_rv.apply(lambda r: any(k in r for k in _bear_kws)).sum())
            _show_cons = True

    col1, col2, col3, col4 = st.columns([2, 1, 1, 3])
    with col1:
        st.subheader(f"{report['company_name']} ({report['ticker']})")
        st.caption(f"{report['sector']} | {report['industry']}")
    with col2:
        st.caption("AI Rating")
        st.subheader(f"{emoji} {report['ai_rating']}")
    with col3:
        st.caption("Current Price")
        st.subheader(f"${report['current_price']:.2f}")
    with col4:
        if _show_cons:
            st.caption("Analyst Consensus")
            _ca, _cb, _cc = st.columns(3)
            _ca.subheader(f"🟢 {_n_bull} Bullish")
            _cb.subheader(f"🟡 {_n_neut} Neutral")
            _cc.subheader(f"🔴 {_n_bear} Bearish")

    st.info(f"**{report['one_line_summary']}**")
    st.caption(f"Rating rationale: {report['ai_rating_rationale']}")
    st.divider()

    # Performance
    st.subheader("Price & Performance")
    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        st.caption("Current Price")
        st.subheader(f"${report['current_price']:.2f}")
    with p2:
        st.caption("1-Year Return")
        st.subheader(f"{report['one_year_return_pct']:.1f}%")
    with p3:
        st.caption("5-Year Return")
        st.subheader(f"{report['five_year_return_pct']:.1f}%")
    with p4:
        st.caption("Market Cap")
        st.subheader(report['market_cap'])
    with p5:
        st.caption("52-Wk Range")
        st.subheader(f"{report['fifty_two_week_low']} – {report['fifty_two_week_high']}")
    st.divider()

    # Chart
    st.subheader("Price History")
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
                "💡 Tip: Toggle **Normalize to 100** in the sidebar for "
                "apples-to-apples comparison across different price levels."
            )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Click legend items to show/hide. "
            "Indices = dotted lines. Double-click to isolate."
        )
    else:
        st.warning("Could not load chart data.")
    st.divider()

    # Fundamentals
    st.subheader("Top 10 Value Investor Fundamentals")
    fund_c1, fund_c2 = st.columns(2)

    _tbl_css = """
    <style>
    .fund-table { width:100%; border-collapse:collapse; font-size:0.875rem; }
    .fund-table th {
        background:#f0f0f0; text-align:left; padding:6px 10px;
        font-weight:600; border-bottom:2px solid #ddd;
    }
    .fund-table td { padding:6px 10px; border-bottom:1px solid #eee; }
    .fund-table tr:last-child td { border-bottom:none; }
    .fund-table td.metric { cursor:help; }
    </style>
    """
    st.markdown(_tbl_css, unsafe_allow_html=True)

    _val_rows = [
        ("1. Trailing P/E",
         "Price-to-Earnings (Trailing): Current stock price divided by earnings per share over the last 12 months (TTM). Lower = cheaper relative to current earnings.",
         report.get('trailing_pe', 'N/A')),
        ("2. Forward P/E",
         "Price-to-Earnings (Forward): Current stock price divided by analyst consensus EPS estimate for the next 12 months. Reflects market expectations for future earnings growth.",
         report.get('forward_pe', 'N/A')),
        ("3. EV/EBITDA",
         "Enterprise Value to EBITDA: Total company value (market cap + debt - cash) divided by earnings before interest, taxes, depreciation and amortization over the last 12 months. Capital-structure-neutral valuation multiple.",
         report.get('ev_ebitda', 'N/A')),
        ("4. Price/Book",
         "Price-to-Book: Current stock price divided by book value per share (assets minus liabilities) as of the most recent quarter. Values below 1.0 suggest the stock trades below net asset value.",
         report.get('price_to_book', 'N/A')),
        ("5. Price/FCF",
         "Price-to-Free Cash Flow: Market capitalization divided by free cash flow over the last 12 months. Measures how many years of free cash flow you are paying for the business.",
         report.get('price_to_fcf', 'N/A')),
    ]
    _qual_rows = [
        ("6. Operating Margin",
         "Operating Margin: Operating income divided by revenue over the last 12 months. Measures what percentage of revenue survives after paying operating costs. Higher = stronger competitive position.",
         report.get('operating_margin', 'N/A')),
        ("7. ROIC",
         "ROIC = NOPAT / Invested Capital, where NOPAT = Operating Income minus taxes paid (tax benefits treated as zero). Invested Capital = Equity + Total Debt - Cash.",
         report.get('roic', 'N/A')),
        ("8. Return on Equity",
         "Return on Equity (ROE): Net income divided by shareholders equity over the last 12 months. Measures how efficiently management generates profit from shareholders' money.",
         report.get('return_on_equity', 'N/A')),
        ("9. Revenue Growth",
         "Revenue Growth (YoY): Year-over-year revenue growth comparing the most recent quarter to the same quarter in the prior year.",
         report.get('revenue_growth_yoy', 'N/A')),
        ("10. Debt/Equity",
         "Debt-to-Equity: Total debt divided by shareholders equity as of the most recent quarter. Measures financial leverage. Higher values mean more debt relative to equity capital.",
         report.get('debt_to_equity', 'N/A')),
        ("11. FCF Yield",
         "Free Cash Flow Yield: Free cash flow over the last 12 months divided by current market capitalization. The cash return you receive on the market price — the inverse of Price/FCF.",
         report.get('fcf_yield', 'N/A')),
    ]

    def _fund_html(header_label, rows, ticker):
        rows_html = "".join(
            f'<tr><td class="metric" title="{tip}">{label}</td><td>{val}</td></tr>'
            for label, tip, val in rows
        )
        return (
            f'<table class="fund-table">'
            f'<thead><tr><th>Metric</th><th>{ticker}</th></tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table>'
        )

    with fund_c1:
        st.markdown("**Valuation**")
        st.markdown(_fund_html("Valuation", _val_rows, report['ticker']),
                    unsafe_allow_html=True)

    with fund_c2:
        st.markdown("**Quality & Growth**")
        st.markdown(_fund_html("Quality & Growth", _qual_rows, report['ticker']),
                    unsafe_allow_html=True)

    # Comps
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
            ("7. ROIC",          "7_roic"),
            ("8. ROE",           "8_return_on_equity"),
            ("9. Rev. Growth",   "9_revenue_growth"),
            ("10. Debt / Equity","10_debt_to_equity"),
            ("11. FCF Yield",    "11_fcf_yield"),
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
            st.dataframe(build_comp_df(metrics_val),
                         hide_index=True, width='stretch')
        with comp_c2:
            st.markdown("**Quality & Growth**")
            st.dataframe(build_comp_df(metrics_qual),
                         hide_index=True, width='stretch')
        st.caption(f"★ = subject company ({report['ticker']})")

    # ── Analyst Recommendations ───────────────────────────────
    has_targets = analyst_targets and any(
        v for v in analyst_targets.values() if v
    )
    has_recs = (analyst_recs is not None and not analyst_recs.empty)

    if has_targets or has_recs:
        st.divider()
        st.subheader("Analyst Recommendations")
        st.caption("Last 3 months | Source: Yahoo Finance | "
                   "Ratings shown as reported by each firm")

        if has_targets:
            current = report['current_price']
            mean_t  = analyst_targets.get("mean")
            high_t  = analyst_targets.get("high")
            low_t   = analyst_targets.get("low")
            med_t   = analyst_targets.get("median")

            def upside_str(val):
                if not val or not current:
                    return None
                pct = ((val - current) / current) * 100
                return f"{pct:+.1f}% vs current"

            a1, a2, a3, a4 = st.columns(4)
            with a1:
                st.caption("Mean Target")
                st.subheader(f"${mean_t:.2f}" if mean_t else "N/A")
                if upside_str(mean_t): st.caption(upside_str(mean_t))
            with a2:
                st.caption("High Target")
                st.subheader(f"${high_t:.2f}" if high_t else "N/A")
                if upside_str(high_t): st.caption(upside_str(high_t))
            with a3:
                st.caption("Low Target")
                st.subheader(f"${low_t:.2f}" if low_t else "N/A")
                if upside_str(low_t): st.caption(upside_str(low_t))
            with a4:
                st.caption("Median Target")
                st.subheader(f"${med_t:.2f}" if med_t else "N/A")
                if upside_str(med_t): st.caption(upside_str(med_t))

            st.caption(
                "Price targets represent the aggregate of **all** price targets "
                "submitted by analysts to Yahoo Finance."
            )

        if has_recs:
            filtered = analyst_recs.copy()
            if selected_firms is not None:
                filtered = filtered[filtered["Firm"].isin(selected_firms)]

            if filtered.empty:
                st.info(
                    "No firms selected. Use the **Analyst Firms** buttons "
                    "in the sidebar to filter."
                )
            else:
                display_df = filtered[
                    ["Date", "Firm", "Rating", "Action"]
                ].copy()
                display_df["Date"] = display_df["Date"].astype(str)

                st.dataframe(display_df, hide_index=True,
                             width='stretch')
        elif has_targets:
            st.info("No individual firm ratings available for this ticker.")

    st.divider()

    # Thesis and risks
    thesis_col, risks_col = st.columns(2)
    with thesis_col:
        st.subheader("Investment Thesis")
        st.write(report['investment_thesis'])
    with risks_col:
        st.subheader("Key Risks")
        for i, risk in enumerate(report['key_risks'], 1):
            st.warning(f"**{i}.** {risk}")

    st.divider()

    # Downloads
    st.subheader("Download Report")
    date_str      = datetime.now().strftime('%Y%m%d')
    filename_stem = f"{report['ticker']}_research_{date_str}"

    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        report_text = build_report_text(
            report, subject_fund, comps_data,
            analyst_targets, analyst_recs, selected_firms
        )
        st.download_button(
            label="⬇️ Download as .txt",
            data=report_text,
            file_name=f"{filename_stem}.txt",
            mime="text/plain",
            width='stretch'
        )

    with dl_col2:
        with st.spinner("Preparing PDF..."):
            pdf_bytes = generate_pdf(
                report, subject_fund, comps_data,
                comp_tickers, chart_period, normalize,
                selected_indices,
                analyst_targets, analyst_recs, selected_firms
            )
        st.download_button(
            label="⬇️ Download as PDF",
            data=pdf_bytes,
            file_name=f"{filename_stem}.pdf",
            mime="application/pdf",
            width='stretch'
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

st.markdown("""
<style>
section[data-testid="stSidebar"] button[kind="primary"] {
    background-color: #000000;
    border-color: #000000;
    color: #ffffff;
}
section[data-testid="stSidebar"] button[kind="primary"]:hover {
    background-color: #333333;
    border-color: #333333;
    color: #ffffff;
}
section[data-testid="stSidebar"] {
    min-width: 340px !important;
}
section[data-testid="stSidebar"] > div:first-child {
    min-width: 340px !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    background-color: #000000;
    color: #ffffff;
    border-radius: 6px;
    font-size: 1.25rem;
    font-weight: 700;
    white-space: nowrap;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    background-color: #333333;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary span {
    color: #ffffff !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    white-space: nowrap !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
    fill: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

_ANALYST_WELCOME = (
    "Hi! I'm your AI analyst. Ask me about any stock, or if you've generated "
    "a report I already have that context loaded."
)

if "analyst_chat" not in st.session_state:
    st.session_state.analyst_chat = [
        {"role": "assistant", "content": _ANALYST_WELCOME}
    ]

with st.sidebar:
    with st.expander("💬 ASK THE AI ASSISTANT", expanded=False):
        if st.button("Clear", key="clear_analyst_chat"):
            st.session_state.analyst_chat = [
                {"role": "assistant", "content": _ANALYST_WELCOME}
            ]
            st.rerun()

        for _msg in st.session_state.analyst_chat:
            with st.chat_message(_msg["role"]):
                st.write(_msg["content"])

        _user_input = st.chat_input(
            "Ask about any stock...", key="analyst_chat_input"
        )

        if _user_input:
            st.session_state.analyst_chat.append(
                {"role": "user", "content": _user_input}
            )

            _base_system = (
                "You are a senior equity research analyst with access to live "
                "market data tools. Answer questions concisely and precisely. "
                "Always cite specific numbers. If a report has been generated, "
                "use that context first before calling tools. "
                "You can fetch data for any ticker symbol using your tools, not "
                "just the current report ticker. If asked to compare stocks, "
                "fetch data for each ticker independently. "
                "Be concise and structured. Use bullet points for financial data. "
                "Lead with the most important facts first. Keep total response "
                "under 400 words unless the user explicitly asks for more detail. "
                "Never leave a response incomplete — if you are running long, "
                "summarize rather than cut off. "
                "When using web search, extract only the key facts needed to "
                "answer the question. Do not reproduce large amounts of web "
                "content. Summarize findings in 2-3 sentences maximum then "
                "provide your analysis."
            )
            if st.session_state.get("report"):
                _r = st.session_state.report
                _ctx = (
                    f"Report context: {_r['ticker']} ({_r['company_name']}), "
                    f"current price ${_r['current_price']:.2f}, "
                    f"AI rating: {_r['ai_rating']}, "
                    f"summary: {_r['one_line_summary']}.\n\n"
                )
                _system = _ctx + _base_system
            else:
                _system = _base_system

            # Build API messages, skipping any leading assistant messages
            _api_msgs = []
            for _m in st.session_state.analyst_chat:
                if not _api_msgs and _m["role"] == "assistant":
                    continue
                _api_msgs.append(
                    {"role": _m["role"], "content": _m["content"]}
                )

            with st.spinner("Thinking..."):
                while True:
                    _resp = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=2048,
                        system=_system,
                        tools=data_tools,
                        messages=_api_msgs,
                    )
                    _api_msgs.append(
                        {"role": "assistant", "content": _resp.content}
                    )
                    if _resp.stop_reason != "tool_use":
                        _reply = next(
                            (b.text for b in _resp.content if b.type == "text"),
                            "",
                        )
                        break
                    _tool_results = []
                    for _blk in _resp.content:
                        if _blk.type == "tool_use":
                            _res = execute_tool(_blk.name, _blk.input)
                            _tool_results.append({
                                "type":        "tool_result",
                                "tool_use_id": _blk.id,
                                "content": (
                                    json.dumps(_res)
                                    if isinstance(_res, dict)
                                    else str(_res)
                                ),
                            })
                    _api_msgs.append(
                        {"role": "user", "content": _tool_results}
                    )

            st.session_state.analyst_chat.append(
                {"role": "assistant", "content": _reply}
            )
            st.rerun()

    st.divider()
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

    if st.session_state.get("report_ready"):
        if st.button("🗑️ Clear Report", use_container_width=True, type="secondary"):
            for _key in ["report", "subject_fund", "comps_data", "comp_tickers",
                         "analyst_targets", "analyst_recs", "all_firms",
                         "selected_firms", "report_ready", "analyst_chat"]:
                st.session_state.pop(_key, None)
            st.rerun()
        st.caption("Clears current report and chat history")

    # Analyst firm buttons — appear whenever firm data is in session state
    if st.session_state.get("all_firms"):

        st.divider()
        st.subheader("Analyst Firms")
        st.caption("Click to toggle firm recommendations")

        sa_col, ca_col = st.columns(2)
        with sa_col:
            if st.button("Select All", use_container_width=True):
                st.session_state.selected_firms = set(
                    st.session_state.all_firms
                )
                st.rerun()
        with ca_col:
            if st.button("Clear All", use_container_width=True):
                st.session_state.selected_firms = set()
                st.rerun()

        for firm in st.session_state.all_firms:
            is_on = firm in st.session_state.selected_firms
            label = f"✓  {firm}" if is_on else f"○  {firm}"
            if st.button(
                label,
                key=f"firm_btn_{firm}",
                type="primary" if is_on else "secondary",
                use_container_width=True
            ):
                if is_on:
                    st.session_state.selected_firms.discard(firm)
                else:
                    st.session_state.selected_firms.add(firm)
                st.rerun()

    st.divider()
    st.caption("Built with Claude + yfinance + Plotly")
    st.caption("Yahoo Finance data — 15-min delay")


# Derived state
selected_indices = []
if show_sp500:
    selected_indices.append("S&P 500")
if show_nasdaq:
    selected_indices.append("NASDAQ")
if show_dow:
    selected_indices.append("Dow Jones")

selected_firms = st.session_state.get("selected_firms", None)


# Main logic
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
                f"**{', '.join(invalid_tickers)}**."
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

        status.update(label="Fetching analyst recommendations...")
        analyst_targets, analyst_recs = get_analyst_data(ticker_input)

        report = generate_research_report(
            ticker_input, history, summary, status
        )

        st.session_state.report          = report
        st.session_state.subject_fund    = subject_fund
        st.session_state.comps_data      = comps_data
        st.session_state.comp_tickers    = comp_tickers
        st.session_state.analyst_targets = analyst_targets
        st.session_state.analyst_recs    = analyst_recs

        all_firms = get_unique_firms(analyst_recs)
        st.session_state.all_firms       = all_firms
        st.session_state.selected_firms  = set(all_firms)
        st.session_state.report_ready    = True

        status.update(label="Report ready!", state="complete")

    # Rerun so the sidebar re-renders with all_firms already in session state,
    # which is required for the analyst firm buttons to appear.
    st.rerun()

elif "report" in st.session_state and not generate_btn:
    st.session_state.report_ready = True
    render_report(
        st.session_state.report,
        st.session_state.subject_fund,
        st.session_state.comps_data,
        st.session_state.comp_tickers,
        chart_period, normalize, selected_indices,
        st.session_state.get("analyst_targets"),
        st.session_state.get("analyst_recs"),
        st.session_state.get("selected_firms"),
    )

else:
    st.info("👈 Enter a ticker in the sidebar and click Generate Report.")