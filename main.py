from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import json
import re
import requests
from datetime import datetime
from collections import Counter
from typing import List, Dict

# Load environment variables from local .env if it exists
if os.path.exists('.env'):
    with open('.env', 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                k, v = line.strip().split('=', 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

GROQ_API_KEY      = os.getenv('GROQ_API_KEY', '')
TAVILY_API_KEY    = os.getenv('TAVILY_API_KEY', '')
META_ACCESS_TOKEN = os.getenv('META_ACCESS_TOKEN', '')
SEARCHAPI_KEY     = os.getenv('SEARCHAPI_KEY', '')

# Apply variables to environment
os.environ['GROQ_API_KEY']   = GROQ_API_KEY
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

CONFIG = {
    'groq_model'         : 'llama-3.3-70b-versatile',
    'temperature'        : 0.2,
    'max_ads_to_fetch'   : 20,
    'max_search_results' : 5,
    'ad_type'             : 'ALL',
    'meta_url'           : 'https://graph.facebook.com/v20.0/ads_archive',
    'cta_keywords'       : [
        'shop now', 'buy now', 'order now', 'daftar', 'coba gratis',
        'pelajari', 'hubungi', 'download', 'daftar sekarang', 'beli sekarang',
        'dapatkan', 'klaim', 'mulai', 'cek', 'lihat'
    ]
}

# In-memory raw ads cache mapping brand/query to list of ad records
raw_ads_cache = {}

# ──────────────────────────────────────────────────────────────────────
# Tool Definitions
# ──────────────────────────────────────────────────────────────────────
from langchain_core.tools import tool
from tavily import TavilyClient

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

@tool
def fetch_meta_ads(search_query: str, country: str = 'ID') -> str:
    """
    Fetch active ads from Meta Ad Library using SearchAPI.io.
    More reliable than direct Meta Graph API for keyword searches.

    Args:
        search_query: Brand name or keyword (e.g. 'Tokopedia', 'cekat ai')
        country: Country code (default: ID for Indonesia)
    """
    try:
        params = {
            'engine'       : 'meta_ad_library',
            'q'            : search_query,
            'country'      : country,
            'ad_type'      : 'all',
            'active_status': 'active',
            'num'          : CONFIG['max_ads_to_fetch'],
            'api_key'      : SEARCHAPI_KEY
        }
        response = requests.get('https://www.searchapi.io/api/v1/search', params=params)
        data     = response.json()

        if 'error' in data:
            return f"SearchAPI Error: {data['error']}"

        ads = data.get('ads', [])
        if not ads:
            return json.dumps({'error': f"No active ads found for '{search_query}'"})

        # Cache for metrics
        raw_ads_cache[search_query] = ads

        # Build clean records for LLM
        records = []
        for ad in ads:
            snap = ad.get('snapshot', {}) or {}
            body = (snap.get('body', {}) or {}).get('text', '') or ''
            records.append({
                'page_name'  : ad.get('page_name', 'Unknown'),
                'is_active'  : ad.get('is_active', False),
                'start_date' : ad.get('start_date', '')[:10],
                'platforms'  : ad.get('publisher_platform', []),
                'body'       : body[:300],
                'title'      : snap.get('title', '') or '',
                'cta_text'   : snap.get('cta_text', '') or '',
                'cta_type'   : snap.get('cta_type', '') or '',
                'media_type' : snap.get('display_format', '') or '',
            })

        return json.dumps({
            'query'        : search_query,
            'country'      : country,
            'total_results': data.get('search_information', {}).get('total_results', len(records)),
            'total_fetched': len(records),
            'ads'          : records
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return f"Error: {str(e)}"


@tool
def fetch_ads_by_page_id(page_id: str, country: str = 'ID') -> str:
    """
    Fetch active ads from a specific page using SearchAPI.io.
    Works with both Facebook Pages and personal/business profiles.

    Args:
        page_id: Facebook Page ID or profile ID (e.g. '61551061527910')
        country : Country code (default: ID)
    """
    try:
        params = {
            'engine'       : 'meta_ad_library',
            'page_id'      : page_id,
            'country'      : country,
            'ad_type'      : 'all',
            'active_status': 'active',
            'num'          : CONFIG['max_ads_to_fetch'],
            'api_key'      : SEARCHAPI_KEY
        }
        response = requests.get('https://www.searchapi.io/api/v1/search', params=params)
        data     = response.json()

        if 'error' in data:
            return f"SearchAPI Error: {data['error']}"

        ads = data.get('ads', [])
        if not ads:
            return json.dumps({'error': f"No active ads found for page_id '{page_id}'"})

        # Cache for metrics
        raw_ads_cache[page_id] = ads

        records = []
        for ad in ads:
            snap = ad.get('snapshot', {}) or {}
            body = (snap.get('body', {}) or {}).get('text', '') or ''
            records.append({
                'page_name' : ad.get('page_name', 'Unknown'),
                'is_active' : ad.get('is_active', False),
                'start_date': ad.get('start_date', '')[:10],
                'platforms' : ad.get('publisher_platform', []),
                'body'      : body[:300],
                'title'     : snap.get('title', '') or '',
                'cta_text'  : snap.get('cta_text', '') or '',
                'cta_type'  : snap.get('cta_type', '') or '',
                'media_type': snap.get('display_format', '') or '',
            })

        return json.dumps({
            'page_id'      : page_id,
            'country'      : country,
            'total_results': data.get('search_information', {}).get('total_results', len(records)),
            'total_fetched': len(records),
            'ads'          : records
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return f"Error: {str(e)}"


@tool
def web_search(query: str) -> str:
    """
    Search the web for additional context about a brand or competitor.
    Use to enrich ad analysis with brand background and marketing strategy.

    Args:
        query: Search query (e.g. 'Tokopedia marketing strategy 2025')
    """
    try:
        results   = tavily_client.search(
            query=query,
            max_results=CONFIG['max_search_results'],
            search_depth='basic'
        )
        formatted = [{
            'title'  : r.get('title', ''),
            'url'    : r.get('url', ''),
            'content': r.get('content', '')[:400]
        } for r in results.get('results', [])]
        return json.dumps({'query': query, 'results': formatted}, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

# ──────────────────────────────────────────────────────────────────────
# Metrics and Normalization
# ──────────────────────────────────────────────────────────────────────
def extract_ad_texts(ads: List[Dict]) -> List[str]:
    texts = []
    for ad in ads:
        parts = (
            ad.get('ad_creative_bodies', []) +
            ad.get('ad_creative_link_titles', []) +
            ad.get('ad_creative_link_descriptions', [])
        )
        text = ' '.join(parts).strip()
        if text:
            texts.append(text)
    return texts

def detect_cta(texts: List[str]) -> Dict:
    cta_counts = Counter()
    ads_with_cta = 0
    for text in texts:
        text_lower = text.lower()
        has_cta = False
        for cta in CONFIG['cta_keywords']:
            if cta in text_lower:
                cta_counts[cta] += 1
                has_cta = True
        if has_cta:
            ads_with_cta += 1
            
    total = len(texts) or 1
    return {
        'cta_frequency'  : dict(cta_counts.most_common(10)),
        'top_cta'        : cta_counts.most_common(1)[0][0] if cta_counts else 'none detected',
        'ads_with_cta_pct': round(ads_with_cta / total * 100, 1)
    }

def analyze_copy_length(texts: List[str]) -> Dict:
    if not texts:
        return {}
    lengths = [len(t.split()) for t in texts]
    short   = sum(1 for l in lengths if l <= 20)
    medium  = sum(1 for l in lengths if 21 <= l <= 60)
    long_   = sum(1 for l in lengths if l > 60)
    total   = len(lengths)
    return {
        'avg_word_count'     : round(sum(lengths) / total, 1),
        'min_word_count'     : min(lengths),
        'max_word_count'     : max(lengths),
        'short_copy_pct'     : round(short  / total * 100, 1),
        'medium_copy_pct'    : round(medium / total * 100, 1),
        'long_copy_pct'      : round(long_  / total * 100, 1),
        'dominant_format'    : 'short' if short >= medium and short >= long_ else
                               'medium' if medium >= long_ else 'long'
    }

def detect_promo_signals(texts: List[str]) -> Dict:
    promo_keywords = [
        'diskon', 'gratis', 'free', 'promo', 'cashback', 'voucher',
        'hemat', 'murah', 'terjangkau', 'flash sale', 'limited',
        '%', 'off', 'ongkir', 'freeship'
    ]
    promo_counts = Counter()
    ads_with_promo = 0
    for text in texts:
        text_lower = text.lower()
        found = False
        for kw in promo_keywords:
            if kw in text_lower:
                promo_counts[kw] += 1
                found = True
        if found:
            ads_with_promo += 1
    total = len(texts) or 1
    return {
        'ads_with_promo_pct' : round(ads_with_promo / total * 100, 1),
        'top_promo_keywords' : dict(promo_counts.most_common(5)),
    }

def compute_metrics(ads: List[Dict]) -> Dict:
    texts       = extract_ad_texts(ads)
    page_names  = [ad.get('page_name', 'Unknown') for ad in ads]
    unique_pages = list(set(page_names))

    cta_analysis   = detect_cta(texts)
    copy_analysis  = analyze_copy_length(texts)
    promo_analysis = detect_promo_signals(texts)

    return {
        'total_ads_analyzed' : len(ads),
        'unique_advertisers' : len(unique_pages),
        'advertiser_names'   : unique_pages[:10],
        'cta_analysis'       : cta_analysis,
        'copy_length'        : copy_analysis,
        'promo_signals'      : promo_analysis,
        'sample_ad_texts'    : texts[:5],
    }

def normalize_ads_for_metrics(ads):
    normalized = []
    for ad in ads:
        if not ad:
            continue
        if 'snapshot' in ad:
            snap  = ad.get('snapshot', {}) or {}
            body  = (snap.get('body', {}) or {}).get('text', '') or ''
            title = snap.get('title', '') or ''
            card_texts = []
            for card in snap.get('cards', []) or []:
                if card.get('body'):
                    card_texts.append(card['body'])
                if card.get('title'):
                    card_texts.append(card['title'])
        else:
            bodies = ad.get('ad_creative_bodies', []) or []
            body   = bodies[0] if bodies else ''
            titles = ad.get('ad_creative_link_titles', []) or []
            title  = titles[0] if titles else ''
            card_texts = ad.get('ad_creative_link_descriptions', []) or []

        normalized.append({
            'page_name'                    : ad.get('page_name', 'Unknown'),
            'ad_creative_bodies'           : [body] if body else [],
            'ad_creative_link_titles'      : [title] if title else [],
            'ad_creative_link_descriptions': [t for t in card_texts if t],
        })
    return normalized

# ──────────────────────────────────────────────────────────────────────
# LangGraph Agent Setup
# ──────────────────────────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

SYSTEM_PROMPT = """
You are AdSpy, a senior paid media strategist with 10+ years experience running
Meta ad campaigns across Southeast Asia. You think like a performance marketer —
every insight must connect to a specific action that improves ROAS or reduces CPL.

WORKFLOW — always follow this order:
1. Fetch active competitor ads using the tools.
   - Use `fetch_ads_by_page_id` if Page ID is provided.
   - Otherwise use `fetch_meta_ads` for keyword/brand search.
2. Use `web_search` to get brand background, positioning, and recent marketing news.
3. Analyze all data thoroughly using the framework below.
4. Produce the full strategic report.

DEEP ANALYSIS FRAMEWORK:

A. CREATIVE ANALYSIS
   - What formats dominate? (single image, video, carousel, UGC-style)
   - What visual hooks are being used? (before/after, testimonial, product demo, lifestyle)
   - What emotions are being triggered? (fear, greed, aspiration, FOMO, curiosity)

B. COPY ANALYSIS
   - What is the dominant messaging angle?
     (promo/discount, pain point, aspiration, social proof, authority, urgency)
   - How is the headline structured? (question, statement, number, command)
   - What specific pain points or desires are being addressed?
   - What power words are frequently used?
   - Copy length pattern: short punchy vs long-form storytelling

C. OFFER ANALYSIS
   - What offers are being promoted? (discount %, free trial, freeship, cashback, bundle)
   - What is the primary CTA? (Shop Now, Learn More, Send Message, Sign Up)
   - Is there a clear value proposition or USP?

D. AUDIENCE SIGNALS
   - What platforms are ads running on? (FB only, IG only, cross-platform)
   - Any signs of retargeting vs cold traffic copy?
   - Language used: formal or casual, Bahasa Indonesia or English?

E. COMPETITIVE GAP ANALYSIS
   - What messaging angles are competitors NOT using?
   - What audience segments appear underserved?
   - What offers or formats are missing from the competitive landscape?

OUTPUT — always produce this exact structure:

## ADSPY INTELLIGENCE REPORT
**Target:** [brand] | **Country:** [country] | **Date:** [date]

### 1. Executive Summary
2-3 sentences capturing the most important competitive insight.

### 2. Ad Audit Overview
- Total active ads, platforms used, estimated activity level
- Key advertisers found (if multiple pages)

### 3. Creative & Messaging Analysis
Break down by angle with REAL examples quoted from actual ad copy.
For each dominant angle explain WHY it works psychologically.

### 4. Offer & CTA Analysis
What offers and CTAs dominate, with specific examples.
What this tells us about their funnel strategy.

### 5. Audience & Platform Strategy
Where they are running, language signals, cold vs warm traffic indicators.

### 6. Competitive Gap Analysis
Specific gaps with explanation of why each is an opportunity.
Prioritize gaps by potential impact (High / Medium / Low).

### 7. Actionable Recommendations
Exactly 5 recommendations. Each must follow this format:

**Recommendation [N]: [Title]**
- **Why:** [What competitor data supports this]
- **What to do:** [Specific, concrete action]
- **Expected impact:** [What metric this should improve and by roughly how much]
- **Priority:** High / Medium / Low

### 8. Quick Win (Do This First)
Single most impactful thing to do in the next 7 days based on the gaps found.

Be ruthlessly specific. Vague insights are useless to a media buyer.
Always reference actual ad copy examples when making claims.
"""

llm = ChatGroq(
    model=CONFIG['groq_model'],
    temperature=CONFIG['temperature'],
    api_key=GROQ_API_KEY,
    streaming=True
)

agent = create_react_agent(
    model=llm,
    tools=[fetch_meta_ads, fetch_ads_by_page_id, web_search],
    prompt=SYSTEM_PROMPT
)

# ──────────────────────────────────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="AdSpy AI Streaming API", version="1.0")

# CORS middleware for open connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze")
async def analyze_competitor_endpoint(request: Request):
    data = await request.json()
    query = data.get("query", "").strip()
    country = data.get("country", "ID").strip()
    mode = data.get("mode", "1").strip() # '1' for Keyword, '2' for Page ID
    
    if not query:
        return {"error": "Query parameter is required."}
        
    async def sse_generator():
        # Clean caches if too big
        if len(raw_ads_cache) > 100:
            raw_ads_cache.clear()
            
        if mode == '2':
            user_message = f"""
            Analyze competitor ads for Page ID: "{query}"
            Target country: {country}
            
            Please:
            1. Fetch active ads using fetch_ads_by_page_id
            2. Search web for brand/competitor context
            3. Generate the full intelligence report
            """
        else:
            user_message = f"""
            Analyze competitor ads for keyword/brand: "{query}"
            Target country: {country}
            
            Please:
            1. Fetch active ads using fetch_meta_ads
            2. Search web for brand/competitor context
            3. Generate the full intelligence report
            """
            
        inputs = {'messages': [HumanMessage(content=user_message)]}
        
        try:
            # Yield initial starting signal
            yield f"event: status\ndata: {json.dumps('🚀 AdSpy intelligence agent initialized...')}\n\n"
            
            async for event in agent.astream_events(inputs, version="v2"):
                kind = event["event"]
                name = event["name"]
                
                # Check for model token streaming
                if kind == "on_chat_model_stream":
                    token = event["data"]["chunk"].content
                    if token:
                        yield f"event: report_chunk\ndata: {json.dumps(token)}\n\n"
                
                # Check for tool call execution
                elif kind == "on_tool_start":
                    tool_input = event["data"].get("input", {})
                    yield f"event: status\ndata: {json.dumps(f'⚙️ Running tool {name} with input: {tool_input}')}\n\n"
                    
                # Check for tool call resolution
                elif kind == "on_tool_end":
                    tool_output = event["data"].get("output", "")
                    output_str = str(tool_output)
                    summary = output_str[:180] + "..." if len(output_str) > 180 else output_str
                    yield f"event: status\ndata: {json.dumps(f'✅ Tool {name} finished. Result: {summary}')}\n\n"
            
            yield f"event: status\ndata: {json.dumps('🏁 Strategic report generation complete!')}\n\n"
            
        except Exception as err:
            yield f"event: error\ndata: {json.dumps(str(err))}\n\n"
            
    return StreamingResponse(sse_generator(), media_type="text/event-stream")


@app.get("/api/proxy-image")
async def proxy_image(url: str):
    """Proxy Facebook CDN images to bypass CORS restrictions."""
    try:
        print(f"Proxying image: {url[:80]}...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Referer': 'https://www.facebook.com/',
        }
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Response status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type')}")
        print(f"Content length: {len(response.content)}")
        
        if response.status_code != 200:
            print(f"Failed with status: {response.status_code}")
            raise HTTPException(status_code=404, detail="Image not found")
        
        return Response(
            content=response.content,
            media_type=response.headers.get('content-type', 'image/jpeg'),
            headers={"Cache-Control": "public, max-age=3600"}
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Proxy error: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search-pages")
async def search_pages_endpoint(q: str, country: str = 'ID'):
    """Helper endpoint to search competitor page IDs by brand keyword."""
    if not q:
        return []
    try:
        params = {
            'engine' : 'meta_ad_library_page_search',
            'q'      : q,
            'country': country,
            'ad_type': 'all',
            'api_key': SEARCHAPI_KEY
        }
        response = requests.get('https://www.searchapi.io/api/v1/search', params=params)
        data = response.json()
        return data.get('page_results', [])
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/metrics")
async def get_metrics_endpoint(query: str):
    """Retrieve computed metrics from the cached raw ads of a request."""
    ads = raw_ads_cache.get(query, [])
    if not ads:
        # Check for matching substring keys in cache
        for key in raw_ads_cache:
            if query.lower() in key.lower() or key.lower() in query.lower():
                ads = raw_ads_cache[key]
                break
                
    if not ads:
        return {"error": f"No cached ad results found for target '{query}'."}
        
    normalized = normalize_ads_for_metrics(ads)
    metrics = compute_metrics(normalized)
    return metrics


@app.get("/api/ads")
async def get_ads_endpoint(query: str):
    """Return up to 20 enriched ad records from cache for display in dashboard."""
    ads = raw_ads_cache.get(query, [])
    if not ads:
        for key in raw_ads_cache:
            if query.lower() in key.lower() or key.lower() in query.lower():
                ads = raw_ads_cache[key]
                break

    if not ads:
        return {"error": f"No cached ads found for '{query}'."}

    result = []
    for ad in ads[:20]:
        snap     = ad.get('snapshot', {}) or {}
        body     = (snap.get('body', {}) or {}).get('text', '') or ''
        title    = snap.get('title', '') or ''
        cta_text = snap.get('cta_text', '') or snap.get('cta_type', '') or ''
        link_url = snap.get('link_url', '') or ''

        # 1. Direct images
        image_url = ''
        images = snap.get('images', []) or []
        if images and isinstance(images, list):
            image_url = images[0].get('original_image_url', '') or images[0].get('resized_image_url', '') or ''

        # 2. Cards (carousel/DPA)
        if not image_url:
            cards = snap.get('cards', []) or []
            for card in cards:
                img = card.get('original_image_url', '') or card.get('resized_image_url', '') or ''
                if img:
                    image_url = img
                    break

        # 3. Video preview thumbnail
        if not image_url:
            videos = snap.get('videos', []) or []
            if videos and isinstance(videos, list):
                image_url = videos[0].get('video_preview_image_url', '') or ''

        result.append({
            'page_name' : ad.get('page_name', 'Unknown'),
            'is_active' : ad.get('is_active', True),
            'start_date': ad.get('start_date', '')[:10] if ad.get('start_date') else '',
            'platforms' : ad.get('publisher_platform', []),
            'body'      : body,
            'title'     : title,
            'cta_text'  : cta_text,
            'link_url'  : link_url,
            'image_url' : image_url,
            'media_type': snap.get('display_format', '') or '',
        })

    return {'total': len(result), 'ads': result}


@app.post("/api/export-pdf")
async def export_pdf_endpoint(request: Request):
    """Render a styled PDF report using WeasyPrint and return as bytes."""
    try:
        import markdown as md_lib
        from weasyprint import HTML

        data    = await request.json()
        report  = data.get("report", "")
        brand   = data.get("brand", "analysis")
        metrics = data.get("metrics")

        # Convert the markdown string into clean HTML tags
        body_html = md_lib.markdown(report, extensions=['tables', 'fenced_code'])

        # Build metrics table if available
        metrics_html = ''
        if metrics:
            cl    = metrics.get('copy_length', {})
            cta   = metrics.get('cta_analysis', {})
            promo = metrics.get('promo_signals', {})
            metrics_html = f"""
            <div class="metrics-box">
                <h3>Metrics Summary</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Ads Analyzed</td><td><b>{metrics.get('total_ads_analyzed', '-')}</b></td></tr>
                    <tr><td>Unique Advertisers</td><td><b>{metrics.get('unique_advertisers', '-')}</b></td></tr>
                    <tr><td>Avg Copy Length</td><td><b>{cl.get('avg_word_count', '-')} words</b></td></tr>
                    <tr><td>Dominant Copy Format</td><td><b>{cl.get('dominant_format', '-')}</b></td></tr>
                    <tr><td>Top CTA</td><td><b>{cta.get('top_cta', '-')}</b></td></tr>
                    <tr><td>Ads with CTA</td><td><b>{cta.get('ads_with_cta_pct', '-')}%</b></td></tr>
                    <tr><td>Ads with Promo Signal</td><td><b>{promo.get('ads_with_promo_pct', '-')}%</b></td></tr>
                </table>
            </div>
            """

        # Construct full HTML with CSS page controls for WeasyPrint
        full_html = f"""<!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                @page {{
                    size: A4;
                    margin: 20mm 15mm;
                }}
                body {{ 
                    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; 
                    font-size: 11pt; 
                    color: #1a1a2e; 
                    margin: 0; 
                    padding: 0; 
                }}
                .header {{ 
                    background: #4361EE; 
                    color: white; 
                    padding: 24px; 
                    margin-bottom: 20px;
                    border-radius: 4px;
                }}
                .header h1 {{ margin: 0 0 6px 0; font-size: 20pt; }}
                .header .sub {{ font-size: 10pt; opacity: 0.9; margin-top: 4px; }}
                h2 {{ font-size: 14pt; color: #4361EE; border-bottom: 2px solid #4361EE; padding-bottom: 4px; margin-top: 24px; page-break-after: avoid; }}
                h3 {{ font-size: 12pt; color: #1a1a2e; margin-top: 16px; page-break-after: avoid; }}
                p, li {{ line-height: 1.6; color: #333; }}
                ul, ol {{ margin-top: 5px; padding-left: 20px; }}
                li {{ margin-bottom: 4px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 10pt; page-break-inside: avoid; }}
                th {{ background: #4361EE; color: white; padding: 8px 12px; text-align: left; }}
                td {{ padding: 8px 12px; border-bottom: 1px solid #e8e8e8; }}
                tr:nth-child(even) td {{ background: #f8f9ff; }}
                .metrics-box {{ background: #f0f4ff; border-left: 4px solid #4361EE; padding: 15px; margin: 20px 0; border-radius: 0 4px 4px 0; page-break-inside: avoid; }}
                .metrics-box h3 {{ margin-top: 0; color: #4361EE; }}
                .footer {{ margin-top: 30px; padding-top: 10px; border-top: 1px solid #e8e8e8; font-size: 9pt; color: #999; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AdSpy Intelligence Report</h1>
                <div class="sub">Target: {brand} | Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}</div>
                <div class="sub">Model: llama-3.3-70b (Groq)</div>
            </div>
            <div class="content">
                {metrics_html}
                {body_html}
            </div>
            <div class="footer">AdSpy Intelligence Agent · Generated via Railway Engine</div>
        </body>
        </html>"""

        # Pass the full compiled HTML string directly to WeasyPrint
        pdf_bytes = HTML(string=full_html).write_pdf()

        # Return the raw binary stream with proper application/pdf content type headers
        safe_brand = re.sub(r'[^a-zA-Z0-9_-]', '_', brand)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=adspy_report_{safe_brand}.pdf"
            }
        )

    except Exception as e:
        print(f"PDF Export Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")




# Serve frontend index.html static file directly at root /
@app.get("/")
async def serve_index():
    return FileResponse("index.html")

# Bind static files directory
app.mount("/static", StaticFiles(directory="."), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)