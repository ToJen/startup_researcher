# startup_research_api.py
"""
FastAPI service that:
1. Accepts a list of startup domains/names
2. Concurrently enriches each company using free‑tier APIs:
   • People Data Labs (PDL)
   • NewsAPI (headlines)
   • Tavily search (optional) – social/funding gap‑fill
   • Firecrawl (optional) – deep crawl for founder bios etc.
3. Persists raw + normalized data in a SQLModel database
4. Exposes:
   • Raw‑JSON REST endpoint per startup
   • `/summary` endpoint with flattened key fields
   • Analytics endpoints (industry counts, top funding, revenue stats)
   • `/chat` endpoint that answers both startup‑specific and analytics questions via OpenAI tool calls

Env vars: PDL_KEY, NEWS_KEY, OPENAI_API_KEY, TVLY_KEY (opt), FC_KEY (opt), DATABASE_URL, LOG_LEVEL
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from firecrawl import FirecrawlApp
from openai import OpenAI
from pydantic import BaseModel
from sqlalchemy import JSON
from sqlmodel import Field, Session, SQLModel, create_engine, select

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("startup_research")

# ---------------------------------------------------------------------------
# Database models
# ---------------------------------------------------------------------------


class Startup(SQLModel, table=True):
    __tablename__ = "startup"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    domain: str = Field(unique=True, index=True)
    name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed: bool = Field(default=False)


class StartupData(SQLModel, table=True):
    __tablename__ = "startupdata"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    startup_id: int = Field(foreign_key="startup.id", index=True)
    key: str  # pdl, news, founders, tavily, firecrawl
    data: str = Field(sa_column=JSON)


engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///startups.db"), echo=False)
SQLModel.metadata.create_all(engine)

# ---------------------------------------------------------------------------
# Helper conversions
# ---------------------------------------------------------------------------


def _funding_to_int(val: Any) -> int:
    try:
        return int(val) if val else 0
    except (TypeError, ValueError):
        return 0


def _revenue_mid(band: str | None) -> int:
    if not band:
        return 0
    nums = [int(n.replace("$", "").replace("M", "000000")) for n in re.findall(r"\$?([0-9]+)M", band)]
    return (sum(nums) // 2) if len(nums) == 2 else (nums[0] if nums else 0)


# ---------------------------------------------------------------------------
# External API helpers (async)
# ---------------------------------------------------------------------------


async def fetch_pdl(client: httpx.AsyncClient, domain: str) -> Dict[str, Any]:
    try:
        logger.info(f"Fetching PDL data for {domain}")
        url = "https://api.peopledatalabs.com/v5/company/enrich"
        params = {"website": domain, "api_key": os.getenv("PDL_KEY")}
        r = await client.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        logger.info(f"Successfully fetched PDL data for {domain}")
        return data
    except Exception as e:
        logger.error(f"Failed to fetch PDL data for {domain}: {e}")
        return {}


async def fetch_pdl_founders(client: httpx.AsyncClient, company_id: str) -> List[Dict[str, Any]]:
    if not company_id:
        return []
    try:
        logger.info(f"Fetching founders for company {company_id}")
        url = "https://api.peopledatalabs.com/v5/person/search"
        body = {
            "query": f'current_company.id:"{company_id}" AND job_title_levels:"owner,founder,cxo"',
            "api_key": os.getenv("PDL_KEY"),
            "size": 25,
            "fields": ["full_name", "linkedin_url", "job_title"],
        }
        r = await client.post(url, json=body, timeout=30)
        r.raise_for_status()
        founders = [
            {"name": p["full_name"], "title": p["job_title"], "linkedin": p.get("linkedin_url")}
            for p in r.json().get("data", [])
        ]
        logger.info(f"Found {len(founders)} founders for company {company_id}")
        return founders
    except Exception as e:
        logger.error(f"Failed to fetch founders for company {company_id}: {e}")
        return []


async def fetch_news(client: httpx.AsyncClient, query: str) -> Dict[str, Any]:
    try:
        logger.info(f"Fetching news for {query}")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 10,
            "apiKey": os.getenv("NEWS_KEY"),
        }
        r = await client.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", [])
        if articles:
            logger.info(f"Found {len(articles)} news articles for {query}")
            return data
        logger.info(f"No news articles found for {query}, using Firecrawl")
    except Exception as e:
        logger.error(f"Failed to fetch news for {query}: {e}")
        logger.info(f"Falling back to Firecrawl for {query}")

    # Firecrawl fallback using firecrawl-py >=2.12.0
    try:
        firecrawl_mod = importlib.util.find_spec("firecrawl")
        if firecrawl_mod is None:
            logger.error("firecrawl-py is not installed. Please install with 'pip install firecrawl-py'.")
            return {"status": "error", "articles": []}
        from firecrawl import FirecrawlApp

        fc_key = os.getenv("FC_KEY")
        if not fc_key:
            logger.error("Firecrawl API key not configured (FC_KEY)")
            return {"status": "error", "articles": []}
        app = FirecrawlApp(api_key=fc_key)
        scrape_result = app.scrape_url(query, formats=["markdown"])
        scrape_result = scrape_result.dict()
        data = scrape_result.get("data", {})
        metadata = data.get("metadata", {})
        title = metadata.get("title", "")
        url = metadata.get("sourceURL", "")
        content = data.get("markdown", "")
        articles = []
        if title or content:
            articles.append(
                {
                    "title": title,
                    "url": url,
                    "content": content,
                }
            )
        logger.info(f"Firecrawl returned {len(articles)} articles for {query}")
        return {"status": "ok", "articles": articles}
    except Exception as e:
        logger.error(f"Firecrawl fallback failed for {query}: {e}")
        return {"status": "error", "articles": []}


async def fetch_tavily(client: httpx.AsyncClient, query: str) -> Dict[str, Any]:
    key = os.getenv("TVLY_KEY")
    if not key:
        logger.info("Tavily API key not configured, skipping")
        return {}
    try:
        logger.info(f"Fetching Tavily data for {query}")
        url = "https://api.tavily.com/search"
        r = await client.post(
            url, json={"query": query, "depth": 3}, headers={"Authorization": f"Bearer {key}"}, timeout=30
        )
        r.raise_for_status()
        data = r.json()
        logger.info(f"Successfully fetched Tavily data for {query}")
        return data
    except Exception as e:
        logger.error(f"Failed to fetch Tavily data for {query}: {e}")
        return {}


async def fetch_firecrawl(client: httpx.AsyncClient, pattern: str) -> dict:
    key = os.getenv("FC_KEY")
    if not key:
        logger.info("Firecrawl API key not configured, skipping")
        return {}
    try:
        logger.info(f"Fetching Firecrawl data for {pattern}")
        app = FirecrawlApp(api_key=key)
        import asyncio

        loop = asyncio.get_event_loop()
        # scrape_url is synchronous, so run in executor
        result = await loop.run_in_executor(None, lambda: app.scrape_url(pattern, formats=["markdown", "html"]))
        result = result.dict()
        logger.info(f"Successfully fetched Firecrawl data for {pattern}")
        return result
    except Exception as e:
        logger.error(f"Failed to fetch Firecrawl data for {pattern}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Enrichment pipeline
# ---------------------------------------------------------------------------


async def enrich_startup(startup_id: int, domain: str):
    try:
        logger.info(f"Starting enrichment for {domain}")
        async with httpx.AsyncClient() as client:
            tasks = {
                "pdl": fetch_pdl(client, domain),
                "news": fetch_news(client, domain),
                "tavily": fetch_tavily(client, f"{domain} linkedin"),
                "firecrawl": fetch_firecrawl(client, f"https://{domain}/*"),
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            payloads = dict(zip(tasks.keys(), results))
            pdl_data = payloads.get("pdl") if not isinstance(payloads.get("pdl"), Exception) else {}
            founders = await fetch_pdl_founders(client, pdl_data.get("id"))
            payloads["founders"] = founders

        with Session(engine) as session:
            for k, v in payloads.items():
                logger.debug(f"Storing key={k}, type={type(v)}")
                if isinstance(v, Exception):
                    continue
                if contains_callable(v):
                    logger.error(f"Payload for key={k} contains a function! Value: {v}")
                    continue
                session.add(StartupData(startup_id=startup_id, key=k, data=json.dumps(v)))
            startup = session.exec(select(Startup).where(Startup.id == startup_id)).one()
            startup.processed = True
            session.commit()
        logger.info(f"Completed enrichment for {domain}")
    except Exception as e:
        logger.error(f"Failed to enrich startup {domain}: {e}")


# ---------------------------------------------------------------------------
# FastAPI app and endpoints
# ---------------------------------------------------------------------------

app = FastAPI(title="Startup Research Assistant")


class StartupsIn(BaseModel):
    domains: List[str]


@app.post("/startups")
async def queue_startups(payload: StartupsIn, bg: BackgroundTasks):
    try:
        logger.info(f"Queueing {len(payload.domains)} startups for enrichment")
        queued = []
        with Session(engine) as session:
            for d in payload.domains:
                dom = d.lower().strip()
                st = session.exec(select(Startup).where(Startup.domain == dom)).first()
                if not st:
                    st = Startup(domain=dom)
                    session.add(st)
                    session.commit()
                    session.refresh(st)
                queued.append(dom)
                bg.add_task(enrich_startup, st.id, dom)
        logger.info(f"Queued {len(queued)} startups")
        return {"queued": queued}
    except Exception as e:
        logger.error(f"Failed to queue startups: {e}")
        raise HTTPException(500, "Failed to queue startups")


@app.get("/startups/{domain}")
async def get_raw(domain: str):
    try:
        logger.info(f"Retrieving raw data for {domain}")
        with Session(engine) as s:
            st = s.exec(select(Startup).where(Startup.domain == domain)).first()
            if not st or not st.processed:
                raise HTTPException(404, "Startup not processed yet")
            blobs = {
                b.key: json.loads(b.data) for b in s.exec(select(StartupData).where(StartupData.startup_id == st.id))
            }
        return blobs
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve raw data for {domain}: {e}")
        raise HTTPException(500, "Failed to retrieve data")


@app.get("/startups/{domain}/summary")
async def get_summary(domain: str):
    try:
        logger.info(f"Retrieving summary for {domain}")
        blobs = await get_raw(domain)  # type: ignore[arg-type]
        pdl = blobs.get("pdl", {})
        return {
            "name": pdl.get("name"),
            "website": pdl.get("website"),
            "location": (pdl.get("location") or {}).get("name"),
            "industry": pdl.get("industry"),
            "revenue_band": pdl.get("inferred_revenue"),
            "amount_raised_usd": (pdl.get("funding") or {}).get("total_funding_usd"),
            "linkedin": pdl.get("linkedin_url"),
            "social_profiles": pdl.get("profiles"),
            "founders": blobs.get("founders", []),
            "recent_news": [a.get("title") for a in blobs.get("news", {}).get("articles", [])],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve summary for {domain}: {e}")
        raise HTTPException(500, "Failed to retrieve summary")


@app.get("/analytics/industry-count")
async def industry_endpoint(min_startups: int = Query(1, ge=1)):
    try:
        logger.info(f"Retrieving industry count analytics (min: {min_startups})")
        return industry_count(min_startups)
    except Exception as e:
        logger.error(f"Failed to retrieve industry count: {e}")
        raise HTTPException(500, "Failed to retrieve industry count")


@app.get("/analytics/top-funded")
async def top_funded_endpoint(limit: int = Query(10, ge=1, le=100)):
    try:
        logger.info(f"Retrieving top funded analytics (limit: {limit})")
        return top_funded(limit)
    except Exception as e:
        logger.error(f"Failed to retrieve top funded analytics: {e}")
        raise HTTPException(500, "Failed to retrieve top funded analytics")


@app.get("/analytics/revenue-stats")
async def revenue_stats_endpoint():
    try:
        logger.info("Retrieving revenue statistics")
        return revenue_stats()
    except Exception as e:
        logger.error(f"Failed to retrieve revenue stats: {e}")
        raise HTTPException(500, "Failed to retrieve revenue statistics")


# ---------------------------------------------------------------------------
# Chat tools
# ---------------------------------------------------------------------------


def db_lookup(company: str) -> str:
    try:
        with Session(engine) as s:
            st = s.exec(select(Startup).where(Startup.domain.contains(company))).first()
            if not st or not st.processed:
                return "I have no data for that company yet."
            blobs = {
                b.key: json.loads(b.data) for b in s.exec(select(StartupData).where(StartupData.startup_id == st.id))
            }
        return str(blobs)
    except Exception as e:
        logger.error(f"Failed to lookup company {company}: {e}")
        return f"Error retrieving data for {company}"


def industry_count(min_startups: int = 1) -> dict:  # reused by REST
    try:
        with Session(engine) as s:
            rows = s.exec(select(StartupData).where(StartupData.key == "pdl")).all()
        cnt: Dict[str, int] = {}
        for r in rows:
            data = json.loads(r.data) if r.data else {}
            ind = data.get("industry") or "Unknown"
            cnt[ind] = cnt.get(ind, 0) + 1
        return {k: v for k, v in cnt.items() if v >= min_startups}
    except Exception as e:
        logger.error(f"Failed to get industry count: {e}")
        return {}


def top_funded(limit: int = 10) -> list:  # reused by REST
    try:
        with Session(engine) as s:
            rows = s.exec(
                select(Startup.domain, StartupData.data)
                .join(StartupData, Startup.id == StartupData.startup_id)
                .where(StartupData.key == "pdl")
            ).all()
        table = [
            {"domain": d, "funding": _funding_to_int(json.loads(data).get("funding", {}).get("total_funding_usd"))}
            for d, data in rows
        ]
        return sorted(table, key=lambda x: x["funding"], reverse=True)[:limit]
    except Exception as e:
        logger.error(f"Failed to get top funded: {e}")
        return []


def revenue_stats() -> dict:  # reused by REST
    try:
        with Session(engine) as s:
            rows = s.exec(select(StartupData.data).where(StartupData.key == "pdl")).all()
        # rows is a list of strings (JSON), not objects with .data
        totals = []
        for r in rows:
            # r is a string (the JSON)
            data = json.loads(r) if isinstance(r, str) else r
            # Try both possible locations for funding
            val = data.get("total_funding_raised") or (data.get("funding") or {}).get("total_funding_usd") or 0
            try:
                val = int(val)
            except Exception:
                val = 0
            totals.append(val)
        return {"average_total_funding_raised": (sum(totals) // len(totals)) if totals else 0, "count": len(totals)}
    except Exception as e:
        logger.error(f"Failed to get revenue stats: {e}")
        return {"average_total_funding_raised": 0, "count": 0}


agent_prompt = (
    "You are a startup‑research assistant. Use the tool functions (`db_lookup`, "
    "`industry_count`, `top_funded`, `revenue_stats`) to answer user questions."
)


@app.post("/chat")
async def chat(user_question: str):
    try:
        logger.info(f"Processing chat request: {user_question[:50]}...")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        messages = [
            {"role": "system", "content": agent_prompt},
            {"role": "user", "content": user_question},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "db_lookup",
                    "description": "Get stored JSON for a company",
                    "parameters": {
                        "type": "object",
                        "properties": {"company": {"type": "string"}},
                        "required": ["company"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "industry_count",
                    "description": "Count startups per industry",
                    "parameters": {
                        "type": "object",
                        "properties": {"min_startups": {"type": "integer"}},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "top_funded",
                    "description": "Return top‑N funded startups",
                    "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "revenue_stats",
                    "description": "Average revenue midpoint across startups",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]

        # First call to get tool calls
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        message = response.choices[0].message
        messages.append(message)

        # Handle tool calls if any
        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                # Call the appropriate function
                if function_name == "db_lookup":
                    function_response = db_lookup(**function_args)
                elif function_name == "industry_count":
                    function_response = industry_count(**function_args)
                elif function_name == "top_funded":
                    function_response = top_funded(**function_args)
                elif function_name == "revenue_stats":
                    function_response = revenue_stats(**function_args)
                else:
                    function_response = "Unknown function"

                # Add the function response to messages
                messages.append({"tool_call_id": tool_call.id, "role": "tool", "content": str(function_response)})

            # Get final response
            final_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )
            answer = final_response.choices[0].message.content
        else:
            answer = message.content

        logger.info("Generated chat response")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Failed to process chat request: {e}")
        raise HTTPException(500, "Failed to process chat request")


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Startup Research Assistant")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# Utility to check for any callable in a nested structure
def contains_callable(obj):
    if callable(obj):
        return True
    if isinstance(obj, dict):
        return any(contains_callable(v) for v in obj.values())
    if isinstance(obj, list):
        return any(contains_callable(i) for i in obj)
    return False
