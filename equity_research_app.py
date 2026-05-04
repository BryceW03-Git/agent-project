import streamlit as st
import streamlit.components.v1 as components
import os
import json
import io
import pandas as pd
from dotenv import load_dotenv
import anthropic
import requests
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
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
client = anthropic.Anthropic(api_key=api_key)

fmp_api_key = os.environ.get("FMP_API_KEY")
if not fmp_api_key:
    try:
        fmp_api_key = st.secrets["FMP_API_KEY"]
    except Exception:
        pass

FMP_BASE = "https://financialmodelingprep.com/stable"


def fmp_get(endpoint, params=None):
    try:
        api_key = os.environ.get("FMP_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets["FMP_API_KEY"]
            except Exception:
                pass
        if not api_key:
            return None
        url = f"{FMP_BASE}{endpoint}"
        p = {"apikey": api_key}
        if params:
            p.update(params)
        r = requests.get(url, params=p, timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


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
        data = fmp_get("/quote", {"symbol": ticker})
        if not data or len(data) == 0:
            return f"Error: No data found for {ticker}"
        q = data[0]
        price = q.get("price", 0)
        from datetime import datetime
        ts = q.get("timestamp")
        date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else "N/A"
        return f"{ticker} most recent price: ${price:.2f} (as of {date})"
    except Exception as e:
        return f"Error fetching {ticker}: {str(e)}"


def get_historical_return(ticker, years):
    try:
        data = fmp_get("/historical-price-eod/light", {"symbol": ticker})
        if not data or len(data) == 0:
            return f"Error: Insufficient data for {ticker}"
        end_price = data[0]["price"]
        end_date = data[0]["date"]
        target_idx = min(len(data) - 1, int(years) * 252)
        start_price = data[target_idx]["price"]
        start_date = data[target_idx]["date"]
        total_return = ((end_price - start_price) / start_price) * 100
        return (f"{ticker} {years}-year return: {total_return:.2f}% "
                f"(${start_price:.2f} to ${end_price:.2f}, "
                f"{start_date} to {end_date})")
    except Exception as e:
        return f"Error computing return for {ticker}: {str(e)}"


def calc_roic_fmp(ticker):
    try:
        income = fmp_get("/income-statement", {"symbol": ticker, "limit": 1})
        balance = fmp_get("/balance-sheet-statement", {"symbol": ticker, "limit": 1})
        if not income or not balance or len(income) == 0 or len(balance) == 0:
            return "N/A"
        inc = income[0]
        bal = balance[0]
        op_income = inc.get("operatingIncome", 0)
        tax_provision = inc.get("incomeTaxExpense", 0)
        equity = bal.get("totalStockholdersEquity", 0)
        debt = bal.get("longTermDebt", 0)
        cash = bal.get("cashAndCashEquivalents", 0)
        if not op_income or not equity:
            return "N/A"
        nopat = op_income - max(0, tax_provision)
        invested_capital = equity + (debt or 0) - (cash or 0)
        if invested_capital <= 0:
            return "N/A"
        roic = (nopat / invested_capital) * 100
        return f"{roic:.1f}%"
    except Exception:
        return "N/A"


def get_fundamentals(ticker):
    try:
        from concurrent.futures import ThreadPoolExecutor
        def _fetch(args):
            endpoint, params = args
            return fmp_get(endpoint, params)

        calls = [
            ("/profile",                  {"symbol": ticker}),
            ("/ratios-ttm",               {"symbol": ticker}),
            ("/quote",                    {"symbol": ticker}),
            ("/key-metrics-ttm",          {"symbol": ticker}),
            ("/income-statement-growth",  {"symbol": ticker, "limit": 1}),
            ("/income-statement",         {"symbol": ticker, "limit": 2}),
            ("/historical-price-eod/light", {"symbol": ticker, "limit": 1}),
        ]
        with ThreadPoolExecutor(max_workers=7) as executor:
            results = list(executor.map(_fetch, calls))

        profile_data, ratios_data, quote_data, km_data, ig_data, is_data, last_price_data = results

        if not profile_data or len(profile_data) == 0:
            return None

        p       = profile_data[0]
        r       = ratios_data[0]  if ratios_data  and len(ratios_data)  > 0 else {}
        q       = quote_data[0]   if quote_data   and len(quote_data)   > 0 else {}
        km      = km_data[0]      if km_data      and len(km_data)      > 0 else {}
        ig      = ig_data[0]      if ig_data      and len(ig_data)      > 0 else {}
        is_list = is_data         if is_data      and len(is_data)      >= 2 else None
        last_trading_date = last_price_data[0].get("date") if last_price_data else None

        if not p.get("companyName"):
            return None

        def pct(val):
            return "N/A" if not val else f"{val * 100:.1f}%"

        def multiple(val, decimals=1):
            return "N/A" if not val else f"{val:.{decimals}f}x"

        def dollar_b(val):
            return "N/A" if not val else f"${val / 1e9:.1f}B"

        roic = calc_roic_fmp(ticker)

        def calc_forward_pe(price, income_statements):
            try:
                if not income_statements or len(income_statements) < 2:
                    return "N/A"

                is0 = income_statements[0]
                is1 = income_statements[1]

                shares0 = is0.get("weightedAverageShsOutDil") or is0.get("weightedAverageShsOut")
                shares1 = is1.get("weightedAverageShsOutDil") or is1.get("weightedAverageShsOut")
                ni0 = is0.get("netIncome")
                ni1 = is1.get("netIncome")

                if not all([shares0, shares1, ni0, ni1]):
                    return "N/A"
                if shares0 <= 0 or shares1 <= 0:
                    return "N/A"

                eps0 = ni0 / shares0
                eps1 = ni1 / shares1

                if eps0 <= 0 or eps1 <= 0:
                    return "N/A"
                if eps1 == 0:
                    return "N/A"

                eps_growth = (eps0 - eps1) / abs(eps1)
                forward_eps = eps0 * (1 + eps_growth)

                if forward_eps <= 0:
                    return "N/A"
                if not price or price <= 0:
                    return "N/A"

                fwd_pe = price / forward_eps
                return f"{fwd_pe:.1f}x"
            except Exception:
                return "N/A"

        return {
            "company_name":        p.get("companyName", ticker),
            "sector":              p.get("sector", "N/A"),
            "industry":            p.get("industry", "N/A"),
            "market_cap":          dollar_b(p.get("mktCap") or q.get("marketCap")),
            "fifty_two_week_high": f"${q.get('yearHigh', 0):.2f}" if q.get("yearHigh") else "N/A",
            "fifty_two_week_low":  f"${q.get('yearLow', 0):.2f}"  if q.get("yearLow")  else "N/A",
            "last_trading_date":   last_trading_date,
            "price_change":        q.get("change"),
            "price_change_pct":    q.get("changePercentage"),

            "1_trailing_pe":       multiple(r.get("priceToEarningsRatioTTM")),
            "2_forward_pe":        calc_forward_pe(q.get("price"), is_list),
            "3_ev_ebitda":         multiple(r.get("enterpriseValueMultipleTTM")),
            "4_price_to_book":     multiple(r.get("priceToBookRatioTTM")),
            "5_price_to_fcf":      multiple(r.get("priceToFreeCashFlowRatioTTM")),
            "6_operating_margin":  pct(r.get("operatingProfitMarginTTM")),
            "7_roic":              roic,
            "8_return_on_equity":  pct(km.get("returnOnEquityTTM")),
            "9_revenue_growth":    pct(ig.get("growthRevenue")),
            "10_debt_to_equity":   multiple(r.get("debtToEquityRatioTTM"), decimals=2),
            "11_fcf_yield":        pct(km.get("freeCashFlowYieldTTM")),
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
    from concurrent.futures import ThreadPoolExecutor
    normalized = [t.upper().strip() for t in comp_tickers]
    def _fetch_comp(t):
        return t, get_fundamentals(t)
    with ThreadPoolExecutor(max_workers=len(normalized) or 1) as executor:
        for t, result in executor.map(_fetch_comp, normalized):
            if result is not None:
                valid[t] = result
            else:
                invalid.append(t)
    return valid, invalid


# ─── ANALYST DATA ────────────────────────────────────────────

def get_analyst_data(ticker):
    try:
        targets = {}
        try:
            pt = fmp_get("/price-target-consensus", {"symbol": ticker})
            if pt and len(pt) > 0:
                t = pt[0]
                targets = {
                    "mean":   t.get("targetConsensus"),
                    "high":   t.get("targetHigh"),
                    "low":    t.get("targetLow"),
                    "median": t.get("targetMedian"),
                }
        except Exception:
            pass

        recs_df = None
        try:
            from datetime import datetime, timedelta
            cutoff = datetime.now() - timedelta(days=90)
            raw = fmp_get("/grades", {"symbol": ticker})
            if raw:
                rows = []
                for item in raw:
                    date_str = item.get("date", "")[:10]
                    if not date_str:
                        continue
                    try:
                        item_date = datetime.strptime(date_str, "%Y-%m-%d")
                    except Exception:
                        continue
                    if item_date < cutoff:
                        continue
                    rows.append({
                        "Date":      date_str,
                        "Firm":      item.get("gradingCompany", "").strip(),
                        "Rating":    item.get("newGrade", "").strip(),
                        "FromGrade": item.get("previousGrade", "").strip(),
                        "Action":    item.get("action", "").strip(),
                    })
                if rows:
                    recs_df = pd.DataFrame(rows)
                    recs_df = recs_df.sort_values("Date", ascending=False).reset_index(drop=True)
        except Exception:
            pass

        return targets, recs_df
    except Exception:
        return {}, None


def get_financial_history(ticker, years=4):
    try:
        years = min(int(years), 10)
        income = fmp_get("/income-statement", {"symbol": ticker, "limit": years})
        if not income:
            return f"No financial history available for {ticker}"
        result = {}
        rows = {
            "Total Revenue":    "revenue",
            "Net Income":       "netIncome",
            "Operating Income": "operatingIncome",
            "Gross Profit":     "grossProfit",
        }
        for label, key in rows.items():
            result[label] = {}
            for stmt in income:
                year = stmt.get("fiscalYear", stmt.get("date", "")[:4])
                val = stmt.get(key)
                if val is not None:
                    result[label][str(year)] = f"${val/1e9:.2f}B"
        latest = income[0] if income else {}
        margins = {}
        rev = latest.get("revenue", 0)
        if rev:
            op = latest.get("operatingIncome", 0)
            net = latest.get("netIncome", 0)
            gross = latest.get("grossProfit", 0)
            if gross: margins["Gross Margin"]     = f"{gross/rev*100:.1f}%"
            if op:    margins["Operating Margin"] = f"{op/rev*100:.1f}%"
            if net:   margins["Net Margin"]       = f"{net/rev*100:.1f}%"
        result["Current Margins"] = margins
        result["Years Shown"] = years
        return result
    except Exception as e:
        return f"Error fetching financial history for {ticker}: {str(e)}"


def get_recent_news(ticker, max_articles=5):
    try:
        max_articles = min(int(max_articles), 10)
        data = fmp_get("/stock_news", {"tickers": ticker, "limit": max_articles})
        if not data:
            return f"No recent news found for {ticker}"
        articles = []
        for item in data[:max_articles]:
            articles.append({
                "title":     item.get("title", "No title"),
                "publisher": item.get("site", "Unknown"),
                "summary":   item.get("text", "")[:300],
                "published": item.get("publishedDate", "")[:10]
            })
        return {
            "ticker":        ticker,
            "article_count": len(articles),
            "articles":      articles
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
            data = fmp_get("/historical-price-eod/light", {"symbol": symbol})
            if data and len(data) > 0:
                import pandas as pd
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                if period == "ytd":
                    df = df[df.index.year == pd.Timestamp.now().year]
                elif period == "1y":
                    df = df[df.index >= pd.Timestamp.now() - pd.DateOffset(years=1)]
                elif period == "3y":
                    df = df[df.index >= pd.Timestamp.now() - pd.DateOffset(years=3)]
                elif period == "5y":
                    df = df[df.index >= pd.Timestamp.now() - pd.DateOffset(years=5)]
                if not df.empty:
                    all_series[label] = df["price"]
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
            data = fmp_get("/historical-price-eod/light", {"symbol": symbol})
            if data and len(data) > 0:
                import pandas as pd
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                if period == "ytd":
                    df = df[df.index.year == pd.Timestamp.now().year]
                elif period == "1y":
                    df = df[df.index >= pd.Timestamp.now() - pd.DateOffset(years=1)]
                elif period == "3y":
                    df = df[df.index >= pd.Timestamp.now() - pd.DateOffset(years=3)]
                elif period == "5y":
                    df = df[df.index >= pd.Timestamp.now() - pd.DateOffset(years=5)]
                if not df.empty:
                    all_series[label] = df["price"]
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

    fig, ax = plt.subplots(figsize=(8.1, 3.4), dpi=150)
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

    plt.tight_layout(pad=0.3, rect=[0, 0, 1, 1])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches=None,
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
    s_oneliner    = style("oneliner",    fontSize=9,
                          fontName="Helvetica-Bold",
                          textColor=colors.HexColor("#222222"), spaceAfter=2)
    s_rationale   = style("rationale",   fontSize=8,
                          textColor=colors.HexColor("#444444"),
                          spaceAfter=8, alignment=TA_JUSTIFY)
    s_bold_body   = style("bold_body",   fontSize=9, leading=14,
                          fontName="Helvetica-Bold", spaceAfter=4)
    s_italic      = style("italic",      fontSize=7.5,
                          fontName="Helvetica-Oblique",
                          textColor=colors.HexColor("#666666"),
                          leading=11, spaceAfter=2, alignment=TA_JUSTIFY)
    s_scenario_hdr = style("scenario_hdr", fontSize=8,
                           fontName="Helvetica-Bold",
                           textColor=colors.HexColor("#111111"),
                           alignment=TA_CENTER, spaceAfter=0)

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
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
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

    # ── Price Target Scenarios ────────────────────────────────
    _cur_p  = report.get('current_price', 0) or 1
    _bear_p = report.get('bear_case_price', 0)
    _base_p = report.get('base_case_price', 0)
    _bull_p = report.get('bull_case_price', 0)

    def _pct_vs(price):
        pct = (price - _cur_p) / _cur_p * 100
        return f"{pct:+.1f}% vs current"

    s_pt_price  = style("pt_price",  fontSize=9,   leading=14,  alignment=TA_CENTER)
    s_pt_pct    = style("pt_pct",    fontSize=7.5, leading=11,
                        textColor=colors.HexColor("#666666"),  alignment=TA_CENTER)
    s_pt_rationale = style("pt_rationale", fontSize=7.5, fontName="Helvetica-Oblique",
                           leading=11, textColor=colors.HexColor("#666666"),
                           alignment=TA_CENTER)

    pt_col_w = W / 3
    pt_tbl = Table(
        [
            [
                Paragraph("Bear Case", s_scenario_hdr),
                Paragraph("Base Case", s_scenario_hdr),
                Paragraph("Bull Case", s_scenario_hdr),
            ],
            [
                Paragraph(f"${_bear_p:.2f}", s_pt_price),
                Paragraph(f"${_base_p:.2f}", s_pt_price),
                Paragraph(f"${_bull_p:.2f}", s_pt_price),
            ],
            [
                Paragraph(_pct_vs(_bear_p), s_pt_pct),
                Paragraph(_pct_vs(_base_p), s_pt_pct),
                Paragraph(_pct_vs(_bull_p), s_pt_pct),
            ],
            [
                Paragraph(report.get('bear_case_rationale', 'N/A'), s_pt_rationale),
                Paragraph(report.get('base_case_rationale', 'N/A'), s_pt_rationale),
                Paragraph(report.get('bull_case_rationale', 'N/A'), s_pt_rationale),
            ],
        ],
        colWidths=[pt_col_w, pt_col_w, pt_col_w]
    )
    pt_tbl.setStyle(TableStyle([
        ("FONTNAME",      (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1,  0), 9),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))

    story.append(KeepTogether([
        Paragraph("PRICE TARGET SCENARIOS", s_section),
        pt_tbl,
        Spacer(1, 6),
        hr(),
    ]))

    # ── Performance ───────────────────────────────────────────
    perf_headers = ["Current Price", "1-Year Return", "5-Year Return",
                    "Market Cap", "52-Week Range"]
    _ch     = subject_fund.get("price_change")     if subject_fund else None
    _ch_pct = subject_fund.get("price_change_pct") if subject_fund else None
    if _ch is not None and _ch_pct is not None:
        _arrow    = "▲" if _ch >= 0 else "▼"
        _ch_color = "#16a34a" if _ch >= 0 else "#dc2626"
        _price_cell = Paragraph(
            f"<b>${report['current_price']:.2f}</b><br/>"
            f"<font size='7' color='{_ch_color}'>{_arrow} ${abs(_ch):.2f} ({_ch_pct:+.2f}%)</font>",
            style("perf_price", fontSize=10, alignment=TA_CENTER)
        )
    else:
        _price_cell = f"${report['current_price']:.2f}"

    perf_values  = [
        _price_cell,
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
        perf_tbl, Spacer(1, 6),
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
    chart_elements += [Spacer(1, 6),
                       hr()]
    story.append(KeepTogether(chart_elements))

    # ── Comps ─────────────────────────────────────────────────
    if comps_data and subject_fund:
        all_tickers = [report['ticker']] + list(comps_data.keys())
        all_data    = {report['ticker']: subject_fund, **comps_data}

        val_metric_keys = [
            ("Trailing P/E",  "1_trailing_pe"),
            ("Forward P/E",   "2_forward_pe"),
            ("EV/EBITDA",     "3_ev_ebitda"),
            ("Price/Book",    "4_price_to_book"),
            ("Price/FCF",     "5_price_to_fcf"),
        ]
        qual_metric_keys = [
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

        def build_comp_rows(keys):
            rows = []
            for label, key in keys:
                row = [label]
                for t in all_tickers:
                    d = all_data.get(t)
                    row.append(d.get(key, "N/A") if d else "N/A")
                rows.append(row)
            return rows

        def build_comp_style(data_full):
            cmds = [
                ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#111111")),
                ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
                ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",      (0, 0), (-1, 0), 8),
                ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                ("ALIGN",         (0, 1), (0, -1),  "LEFT"),
                ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE",      (0, 1), (-1, -1), 8),
                ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
                ("TOPPADDING",    (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING",   (0, 0), (-1, -1), 4),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
                ("FONTNAME",      (1, 1), (1, -1), "Helvetica-Bold"),
            ]
            for i in range(2, len(data_full), 2):
                cmds.append(
                    ("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f9f9f9"))
                )
            return cmds

        n_comps    = len(all_tickers)
        half_w     = W / 2
        metric_col = 0.75 * inch if n_comps >= 5 else 0.9 * inch
        ticker_col = (half_w - metric_col) / n_comps
        col_widths = [metric_col] + [ticker_col] * n_comps

        val_data  = [comp_header] + build_comp_rows(val_metric_keys)
        qual_data = [comp_header] + build_comp_rows(qual_metric_keys)

        val_comp_tbl  = Table(val_data,  colWidths=col_widths)
        qual_comp_tbl = Table(qual_data, colWidths=col_widths)
        val_comp_tbl.setStyle(TableStyle(build_comp_style(val_data)))
        qual_comp_tbl.setStyle(TableStyle(build_comp_style(qual_data)))

        s_comp_lbl = style("comp_lbl", fontName="Helvetica-Bold",
                           fontSize=9, spaceBefore=4, spaceAfter=4)

        comp_layout = Table(
            [
                [Paragraph("Valuation", s_comp_lbl),
                 Paragraph("Quality &amp; Growth", s_comp_lbl)],
                [val_comp_tbl, qual_comp_tbl],
            ],
            colWidths=[W / 2, W / 2],
        )
        comp_layout.setStyle(TableStyle([
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING",   (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
        ]))

        story.append(KeepTogether([
            Paragraph("TOP 10 VALUE INVESTOR FUNDAMENTALS &amp; COMPARABLE ANALYSIS", s_section),
            comp_layout,
            Paragraph(f"* = subject company ({report['ticker']})", s_small),
            Spacer(1, 6),
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
            apt_col_w    = W / len(pt_headers)
            analyst_pt_tbl = Table([pt_headers, pt_values],
                                   colWidths=[apt_col_w] * len(pt_headers))
            analyst_pt_tbl.setStyle(header_table_style(2))
            analyst_pt_tbl.setStyle(TableStyle([("ALIGN", (0, 1), (0, -1), "CENTER")]))
            analyst_elements.append(analyst_pt_tbl)
            analyst_elements.append(Paragraph(
                "Price targets represent the aggregate of all price targets "
                "submitted by analysts to Financial Modeling Prep.",
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
                    "Last 3 months only. Source: Financial Modeling Prep.",
                    s_small
                ))

        analyst_elements += [
            Spacer(1, 6),
            hr(),
        ]
        story.append(KeepTogether(analyst_elements))

    # ── Thesis ────────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph("INVESTMENT THESIS", s_section),
        Paragraph(report.get('investment_thesis', 'N/A'), s_body),
        Spacer(1, 6),
        hr(),
    ]))

    # ── Moat Analysis ─────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph("MOAT ANALYSIS", s_section),
        Paragraph(f"<b>Moat Rating: {report.get('moat_rating', 'N/A')}</b>", s_body),
        Paragraph(f"<b>Moat Description:</b> {report.get('moat_exists', 'N/A')}", s_body),
        Paragraph(f"<b>Sustainability:</b> {report.get('moat_sustainability', 'N/A')}", s_body),
        Paragraph(f"<b>Competitive threats:</b> {report.get('moat_risks', 'N/A')}", s_body),
        Spacer(1, 6),
        hr(),
    ]))

    # ── Upside Catalysts & Key Risks ──────────────────────────
    def _shaded_row(text, bg_hex, border_hex):
        tbl = Table([[Paragraph(text, s_body)]], colWidths=[W])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, 0), colors.HexColor(bg_hex)),
            ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor(border_hex)),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ]))
        tbl.spaceAfter = 3
        return tbl

    risks_elements = [Paragraph("UPSIDE CATALYSTS &amp; KEY RISKS", s_section)]
    risks_elements.append(Paragraph("Upside Catalysts:", s_bold_body))
    for i, cat in enumerate(report.get('upside_catalysts', []), 1):
        risks_elements.append(_shaded_row(f"{i}. {cat}", "#D4EDDA", "#B2DFDB"))
    risks_elements.append(Spacer(1, 6))
    risks_elements.append(Paragraph("Key Risks:", s_bold_body))
    for i, risk in enumerate(report.get('key_risks', []), 1):
        risks_elements.append(_shaded_row(f"{i}. {risk}", "#FFDEDE", "#FFCDD2"))
    risks_elements += [Spacer(1, 6), hr()]
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
            temperature=0,
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
                if status:
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
            "upside_catalysts": {
                "type":  "array",
                "items": {"type": "string"}
            },
            "one_line_summary": {"type": "string"},
            "moat_rating": {
                "type": "string",
                "enum": ["Wide Moat", "Narrow Moat", "No Moat"]
            },
            "moat_exists":        {"type": "string"},
            "moat_sustainability": {"type": "string"},
            "moat_risks":          {"type": "string"},
            "bear_case_price":     {"type": "number"},
            "bear_case_rationale": {"type": "string"},
            "base_case_price":     {"type": "number"},
            "base_case_rationale": {"type": "string"},
            "bull_case_price":     {"type": "number"},
            "bull_case_rationale": {"type": "string"}
        },
        "required": [
            "ticker", "company_name", "sector", "industry",
            "current_price", "fifty_two_week_high", "fifty_two_week_low",
            "one_year_return_pct", "five_year_return_pct", "market_cap",
            "trailing_pe", "forward_pe", "ev_ebitda",
            "price_to_book", "price_to_fcf", "operating_margin",
            "roic", "return_on_equity", "revenue_growth_yoy", "debt_to_equity",
            "fcf_yield", "ai_rating", "ai_rating_rationale",
            "investment_thesis", "key_risks", "upside_catalysts", "one_line_summary",
            "moat_rating", "moat_exists", "moat_sustainability", "moat_risks",
            "bear_case_price", "bear_case_rationale",
            "base_case_price", "base_case_rationale",
            "bull_case_price", "bull_case_rationale"
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
    - One-line summary under 20 words.
    - Moat rating must be Wide Moat, Narrow Moat, or No Moat. Base it on evidence from the fundamentals — high ROIC (>15%), high operating margins (>20%), and consistent revenue growth are signals of a moat. Cite the specific metric that supports the rating.
    - moat_exists: identify which of the five moat types applies (cost advantage, network effects, switching costs, intangible assets, efficient scale) or state none exist.
    - moat_sustainability: assess durability over 5-10 years. Consider industry disruption risk, competitive intensity, and regulatory environment.
    - moat_risks: name the specific competitors or structural forces that threaten the moat today.
    - Bear/Base/Bull price targets must be grounded in the actual current price and fundamentals already gathered. Use the forward P/E as the primary valuation anchor.
    - Bear case: apply a multiple compression or earnings reduction scenario consistent with the key risks identified. Must be below current price.
    - Base case: apply current forward P/E to consensus earnings estimate. Should be close to current analyst mean price target if available.
    - Bull case: apply multiple expansion or earnings acceleration scenario consistent with the upside catalysts identified. Must be above current price.
    - Each rationale is one sentence maximum explaining the key assumption driving that price target.
    - Express all three as specific dollar amounts, not ranges.
    - Exactly 3 upside catalysts, one crisp sentence each. These should be specific near-term or medium-term events or factors that could drive the stock toward the bull case price target — earnings beats, product launches, market share gains, margin expansion, M&A, regulatory approvals, macro tailwinds. Do not repeat what is already stated in the bull case rationale — add specificity and depth.
    - Exactly 3 key risks, one crisp sentence each. These should be specific factors that could drive the stock toward the bear case price target. Do not repeat what is already stated in the bear case rationale — add specificity and depth.
    - Upside catalysts and key risks should complement the Bear/Base/Bull scenarios already in the report, not duplicate them."""

    messages = conversation_history + [{
        "role": "user",
        "content": (
            f"Produce a structured research report for {ticker}.\n\n"
            f"Data summary: {data_summary}"
        )
    }]

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        temperature=0,
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
  PRICE TARGET SCENARIOS
{thin}"""

    _cur_txt = r.get('current_price', 0) or 1
    for _label, _price_key, _rat_key in [
        ("Bear Case", "bear_case_price", "bear_case_rationale"),
        ("Base Case", "base_case_price", "base_case_rationale"),
        ("Bull Case", "bull_case_price", "bull_case_rationale"),
    ]:
        _p   = r.get(_price_key, 0)
        _pct = (_p - _cur_txt) / _cur_txt * 100
        _rat = r.get(_rat_key, 'N/A')
        output += f"\n  {_label:<12}${_p:.2f}  ({_pct:+.1f}% vs current)"
        output += f"\n  {'':12}{_rat}\n"

    output += f"""
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
        output += f"\n{div}\n  TOP 10 VALUE INVESTOR FUNDAMENTALS & COMPARABLE ANALYSIS"
        output += comp_block

    if analyst_targets and any(v for v in analyst_targets.values() if v):
        current = r['current_price']
        mean_t  = analyst_targets.get("mean")
        output += f"\n{div}\n  ANALYST RECOMMENDATIONS\n{thin}"
        output += (
            "\n  Price targets represent the aggregate of all price targets "
            "submitted by analysts to Financial Modeling Prep.\n"
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
  {r.get('investment_thesis', 'N/A')}
{div}
  MOAT ANALYSIS
{thin}
  Moat Rating: {r.get('moat_rating', 'N/A')}

  Moat Description: {r.get('moat_exists', 'N/A')}

  Sustainability: {r.get('moat_sustainability', 'N/A')}

  Competitive threats: {r.get('moat_risks', 'N/A')}
{div}
  UPSIDE CATALYSTS & KEY RISKS
{thin}
  Upside Catalysts:"""
    for i, cat in enumerate(r.get('upside_catalysts', []), 1):
        output += f"\n  {i}. {cat}"
    output += f"\n\n  Key Risks:"
    for i, risk in enumerate(r.get('key_risks', []), 1):
        output += f"\n  {i}. {risk}"
    output += f"""
{div}
  DISCLAIMER: AI-generated. Not investment advice.
  Verify all data independently before making decisions.
{div}"""
    return output


METRIC_TOOLTIPS = {
    "1. Trailing P/E":   "Trailing P/E: Current price divided by the last 12 months of actual earnings per share. Lower = cheaper relative to current earnings.",
    "2. Forward P/E":    "Forward P/E: Current price divided by estimated next-year EPS. Estimated EPS applies the most recent year-over-year EPS growth rate to the most recent annual EPS figure. Lower = cheaper relative to expected earnings.",
    "3. EV / EBITDA":    "EV/EBITDA: Enterprise Value divided by Earnings Before Interest, Taxes, Depreciation & Amortization. Useful for comparing companies with different capital structures.",
    "4. Price / Book":   "Price/Book: Market price divided by book value per share. A ratio below 1.0 means the stock trades below net asset value.",
    "5. Price / FCF":    "Price/FCF: Market cap divided by Free Cash Flow. Measures how much you pay for each dollar of cash the business generates.",
    "6. Oper. Margin":   "Operating Margin: Operating Income divided by Revenue. Measures what percentage of revenue becomes operating profit after operating expenses.",
    "7. ROIC":           "ROIC: Return on Invested Capital = NOPAT / (Equity + Debt - Cash). Measures how efficiently management deploys capital. Above 15% is excellent.",
    "8. ROE":            "ROE: Return on Equity = Net Income / Shareholders Equity. Measures profitability relative to shareholder investment.",
    "9. Rev. Growth":    "Revenue Growth: Year-over-year percentage change in total revenue. Measures top-line business momentum.",
    "10. Debt / Equity": "Debt/Equity: Total Debt divided by Total Shareholders Equity. Measures financial leverage. Higher = more debt relative to equity.",
    "11. FCF Yield":     "FCF Yield: Free Cash Flow divided by Market Cap. The inverse of Price/FCF. Higher = more cash generated relative to market price.",
}

# ─── HTML TABLE WITH INLINE HOVER TOOLTIPS ───────────────────

def build_html_table(rows, columns, table_id="table"):
    SPAN_STYLE = "cursor:default;"
    HOVER_ON  = """
var tt = document.getElementById('shared-tt');
var r  = this.getBoundingClientRect();
var tw = 260, th = 150;
var l  = r.left;
var t  = r.bottom + 4;
if (l + tw > window.innerWidth - 8) l = window.innerWidth - tw - 8;
if (l < 8) l = 8;
if (t + th > window.innerHeight - 8) t = r.top - th - 4;
if (t < 8) t = 8;
tt.innerHTML = this.dataset.tip;
tt.style.left = l + 'px';
tt.style.top  = t + 'px';
tt.style.visibility = 'visible';
""".replace('\n', ' ')
    HOVER_OFF = "document.getElementById('shared-tt').style.visibility='hidden';"

    def make_span(display, tooltip_text):
        if not tooltip_text:
            return str(display)
        safe_tip = tooltip_text.replace("'", "&#39;").replace('"', '&quot;')
        return (
            f"<span style='{SPAN_STYLE}' "
            f"data-tip='{safe_tip}' "
            f"onmouseover=\"{HOVER_ON}\" "
            f"onmouseout=\"{HOVER_OFF}\">"
            f"{display}"
            f"</span>"
        )

    shared_tooltip = """
<div id='shared-tt' style='
    position:fixed;
    visibility:hidden;
    z-index:9999;
    background:#333333;
    color:#e0e0e0;
    border:1px solid #555;
    border-radius:6px;
    padding:8px 10px;
    width:260px;
    white-space:normal;
    font-size:13px;
    font-family:inherit;
    line-height:1.5;
    box-shadow:0 4px 12px rgba(0,0,0,0.5);
    pointer-events:none;
'></div>
"""

    html = (
        shared_tooltip
        + "<style>@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600&display=swap');"
        "body { margin:0; padding:0; }"
        "* { font-family: 'Source Sans Pro', sans-serif; font-size: 14px; color: #31333f; }</style>"
        "<div style='border:1px solid rgba(49,51,63,0.2); border-radius:4px; overflow:hidden;'>"
        f"<table id='{table_id}' style='width:100%; border-collapse:collapse; background:transparent;'>"
    )
    html += "<thead><tr>"
    for i, col in enumerate(columns):
        align = "left" if i == 0 else "center"
        html += f"<th style='text-align:{align}; padding:6px 8px; background:#fafbfc; border-bottom:1px solid rgba(49,51,63,0.1); color:#808495; font-weight:500;'>{col}</th>"
    html += "</tr></thead><tbody>"

    for i, row in enumerate(rows):
        bg = "transparent"
        html += f"<tr style='background:{bg};'>"
        for j, col in enumerate(columns):
            align = "left" if j == 0 else "center"
            if j == 0:
                display      = row.get("metric", col)
                tooltip_text = row.get("tooltip", "")
            else:
                display      = row.get(col, "—")
                tooltip_text = row.get(col + "_cell_tooltip", "")
            cell_content = make_span(display, tooltip_text)
            html += f"<td style='text-align:{align}; padding:6px 8px; border-bottom:1px solid rgba(0,0,0,0.07); overflow:visible; color:#31333f;'>{cell_content}</td>"
        html += "</tr>"

    html += "</tbody></table></div>"
    return html


# ─── RENDER FUNCTION ─────────────────────────────────────────

def render_report(report, subject_fund, comps_data, comp_tickers,
                  chart_period, normalize, selected_indices=None,
                  analyst_targets=None, analyst_recs=None,
                  selected_firms=None):

    st.markdown("""
<style>
div[data-testid="column"] [data-testid="stCaptionContainer"] {
    margin-bottom: -0.6rem;
}
</style>
""", unsafe_allow_html=True)

    elapsed = st.session_state.get("_report_elapsed")
    if elapsed:
        col_spacer, col_timer = st.columns([8, 2])
        with col_timer:
            st.markdown(
                f"<div style='text-align:right; color:#9ca3af; font-size:0.75rem;'>"
                f"⏱ Generated in {elapsed}s</div>",
                unsafe_allow_html=True
            )

    period_code = PERIOD_MAP[chart_period]
    st.divider()

    # Rating emoji helper
    rating_colors = {
        "Strong Buy": "🟢", "Buy": "🟢",
        "Hold": "🟡", "Sell": "🔴", "Strong Sell": "🔴",
    }
    ai_emoji = rating_colors.get(report.get('ai_rating', ''), "⚪")

    # Analyst consensus counts (default 0 when no data)
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

    # ROW 1 — company info
    st.subheader(f"{report['company_name']} ({report['ticker']})  —  ${report['current_price']:.2f}")
    st.caption(f"{report['sector']} | {report['industry']}")

    # ROW 2 — rating lights: emoji on top, label below
    def _rating_cell(emoji, subtext):
        return (
            f"<div style='text-align:center'>"
            f"<p style='font-size:1.5rem;margin:0;line-height:1.1'>{emoji}</p>"
            f"<p style='font-size:0.85rem;font-weight:600;margin:3px 0 0 0'>{subtext}</p>"
            f"</div>"
        )

    stat_left, sep_col, stat_right = st.columns([10, 1, 30])
    with stat_left:
        st.markdown(
            "<p style='text-align:center;font-size:0.75rem;color:#6b7280;"
            "margin:0 0 4px 0'>AI Rating</p>",
            unsafe_allow_html=True
        )
        st.markdown(_rating_cell(ai_emoji, report.get('ai_rating', '—')),
                    unsafe_allow_html=True)
    with sep_col:
        st.markdown(
            "<div style='width:1px;background:#e5e7eb;height:72px;margin:0 auto'></div>",
            unsafe_allow_html=True
        )
    with stat_right:
        st.markdown(
            "<p style='text-align:center;font-size:0.75rem;color:#6b7280;"
            "margin:0 0 4px 0'>Analyst Ratings</p>",
            unsafe_allow_html=True
        )
        _ca, _cb, _cc = st.columns(3)
        _ca.markdown(_rating_cell("🟢", f"{_n_bull} Bullish"), unsafe_allow_html=True)
        _cb.markdown(_rating_cell("🟡", f"{_n_neut} Neutral"), unsafe_allow_html=True)
        _cc.markdown(_rating_cell("🔴", f"{_n_bear} Bearish"), unsafe_allow_html=True)

    st.info(
        f"- {report.get('one_line_summary', 'N/A')}\n"
        f"- Rating Rationale: {report.get('ai_rating_rationale', 'N/A')}"
    )
    st.divider()

    # Price Target Scenarios
    st.subheader("Price Target Scenarios")
    _cur = report.get('current_price', 0) or 1
    _bear_p = report.get('bear_case_price', 0)
    _base_p = report.get('base_case_price', 0)
    _bull_p = report.get('bull_case_price', 0)

    pt1, pt2, pt3 = st.columns(3)
    pt1.metric(
        label="Bear Case",
        value=f"${_bear_p:.2f}",
        delta=f"{((_bear_p - _cur) / _cur * 100):.1f}% vs current"
    )
    pt2.metric(
        label="Base Case",
        value=f"${_base_p:.2f}",
        delta=f"{((_base_p - _cur) / _cur * 100):.1f}% vs current"
    )
    pt3.metric(
        label="Bull Case",
        value=f"${_bull_p:.2f}",
        delta=f"{((_bull_p - _cur) / _cur * 100):.1f}% vs current"
    )

    rc1, rc2, rc3 = st.columns(3)
    rc1.caption(f"_{report.get('bear_case_rationale', 'N/A')}_")
    rc2.caption(f"_{report.get('base_case_rationale', 'N/A')}_")
    rc3.caption(f"_{report.get('bull_case_rationale', 'N/A')}_")
    st.divider()

    # Performance
    st.subheader("Price & Performance")
    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        change = subject_fund.get("price_change") if subject_fund else None
        change_pct = subject_fund.get("price_change_pct") if subject_fund else None
        if change is not None and change_pct is not None:
            delta_str = f"-${abs(change):.2f} ({change_pct:+.2f}%)" if change < 0 else f"${abs(change):.2f} ({change_pct:+.2f}%)"
            st.metric(
                label="Current Price",
                value=f"${report['current_price']:.2f}",
                delta=delta_str
            )
        else:
            st.metric(
                label="Current Price",
                value=f"${report['current_price']:.2f}"
            )
        last_date = subject_fund.get("last_trading_date") if subject_fund else None
        if last_date:
            from datetime import datetime
            parsed_date = datetime.strptime(last_date, "%Y-%m-%d")
            st.caption(f"Trading Date: {parsed_date.strftime('%m/%d/%y')} | 15-min delayed price | Source: FMP")
        else:
            st.caption("15-min delayed price | Source: FMP")
    with p2:
        st.metric(label="1-Year Return", value=f"{report['one_year_return_pct']:.1f}%")
    with p3:
        st.metric(label="5-Year Return", value=f"{report['five_year_return_pct']:.1f}%")
    with p4:
        st.metric(label="Market Cap", value=report['market_cap'])
    with p5:
        st.metric(label="52-Week Range", value=f"{report.get('fifty_two_week_low', 'N/A')} – {report.get('fifty_two_week_high', 'N/A')}")
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

    # Comps
    if True:
        if comps_data:
            st.subheader("Top 10 Value Investor Fundamentals & Comparable Analysis")
        else:
            st.subheader("Top 10 Value Investor Fundamentals")
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

        def build_html_rows(metrics):
            rows = []
            for display_name, key in metrics:
                row = {
                    "metric": display_name,
                    "tooltip": METRIC_TOOLTIPS.get(display_name, "")
                }
                for t in all_tickers:
                    label = f"★ {t}" if t == report['ticker'] else t
                    d = all_data.get(t)
                    val = d.get(key, "N/A") if d else "N/A"
                    row[label] = val
                    cell_tip = d.get(key + "_cell_tooltip", "") if d else ""
                    row[label + "_cell_tooltip"] = cell_tip
                rows.append(row)
            return rows

        comp_c1, comp_c2 = st.columns(2)

        all_tickers_labeled = [f"★ {t}" if t == report['ticker'] else t for t in all_tickers]
        val_columns = ["Metric"] + all_tickers_labeled
        qual_columns = ["Metric"] + all_tickers_labeled

        with comp_c1:
            st.markdown("**Valuation**")
            val_rows = build_html_rows(metrics_val)
            val_height = 36 + (len(val_rows) * 33)
            components.html(build_html_table(val_rows, val_columns, "val_table"), height=val_height, scrolling=False)

        with comp_c2:
            st.markdown("**Quality & Growth**")
            qual_rows = build_html_rows(metrics_qual)
            qual_height = 36 + (len(qual_rows) * 33)
            components.html(build_html_table(qual_rows, qual_columns, "qual_table"), height=qual_height, scrolling=False)

        if comps_data:
            st.caption(f"★ = subject company ({report['ticker']})")

    # ── Analyst Recommendations ───────────────────────────────
    has_targets = analyst_targets and any(
        v for v in analyst_targets.values() if v
    )
    has_recs = (analyst_recs is not None and not analyst_recs.empty)

    if has_targets or has_recs:
        st.divider()
        st.subheader("Analyst Recommendations")
        st.caption("Last 3 months | Source: Financial Modeling Prep | "
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
                st.metric(
                    label="Mean Target",
                    value=f"${mean_t:.2f}" if mean_t else "N/A",
                    delta=upside_str(mean_t) if mean_t else None
                )
            with a2:
                st.metric(
                    label="High Target",
                    value=f"${high_t:.2f}" if high_t else "N/A",
                    delta=upside_str(high_t) if high_t else None
                )
            with a3:
                st.metric(
                    label="Low Target",
                    value=f"${low_t:.2f}" if low_t else "N/A",
                    delta=upside_str(low_t) if low_t else None
                )
            with a4:
                st.metric(
                    label="Median Target",
                    value=f"${med_t:.2f}" if med_t else "N/A",
                    delta=upside_str(med_t) if med_t else None
                )

            st.caption(
                "Price targets represent the aggregate of **all** price targets "
                "submitted by analysts to Financial Modeling Prep."
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

    # Investment Thesis (full width)
    st.subheader("Investment Thesis")
    st.write(report.get('investment_thesis', 'N/A'))
    st.divider()

    # Moat Analysis (full width)
    st.subheader("Moat Analysis")
    st.write(f"**Moat Description:** {report.get('moat_exists', 'N/A')}")
    st.write(f"**Sustainability:** {report.get('moat_sustainability', 'N/A')}")
    st.write(f"**Competitive threats:** {report.get('moat_risks', 'N/A')}")
    st.divider()

    # Upside Catalysts & Key Risks (full width)
    st.subheader("Upside Catalysts & Key Risks")

    st.markdown("**Upside Catalysts**")
    for i, catalyst in enumerate(report.get('upside_catalysts', []), 1):
        st.success(f"🟢 {i}. {catalyst}")

    st.write("")

    st.markdown("**Key Risks**")
    for i, risk in enumerate(report.get('key_risks', []), 1):
        st.error(f"🔴 {i}. {risk}")

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
    st.caption("Built with Claude + FMP + Plotly")
    st.caption("Data provided by Financial Modeling Prep")


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

    import time
    _report_start_time = time.time()

    with st.status(
        f"Researching {ticker_input}...", expanded=True
    ) as status:

        from concurrent.futures import ThreadPoolExecutor

        status.update(label="Gathering market data and fetching comps...")

        def _run_market_data():
            return gather_market_data(ticker_input, None)

        def _run_comps():
            if comps_input.strip():
                return get_comps_data(raw_comps)
            return {}, []

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_market = executor.submit(_run_market_data)
            future_comps  = executor.submit(_run_comps)
            history, summary = future_market.result()
            comps_data, invalid_tickers = future_comps.result()

        if invalid_tickers:
            st.warning(f"⚠️ Skipped invalid ticker(s): **{', '.join(invalid_tickers)}**.")

        comp_tickers = list(comps_data.keys())

        status.update(label="Fetching subject fundamentals...")
        subject_fund = get_fundamentals(ticker_input)
        if not isinstance(subject_fund, dict):
            subject_fund = None


        status.update(label="Fetching analyst recommendations...")
        analyst_targets, analyst_recs = get_analyst_data(ticker_input)

        report = generate_research_report(
            ticker_input, history, summary, status
        )

        if subject_fund and isinstance(subject_fund, dict):
            computed_overrides = {
                'market_cap':          subject_fund.get('market_cap',          report.get('market_cap',          'N/A')),
                'fifty_two_week_high': subject_fund.get('fifty_two_week_high', report.get('fifty_two_week_high', 'N/A')),
                'fifty_two_week_low':  subject_fund.get('fifty_two_week_low',  report.get('fifty_two_week_low',  'N/A')),
            }
            report.update(computed_overrides)

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
        st.session_state["_report_elapsed"] = round(time.time() - _report_start_time, 1)

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