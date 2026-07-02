"""SocialIntel Service

SocialIntel is the ecosystem's Social Intelligence Service (non-trading, non-strategy).

This repo currently contains a placeholder. This file implements an MVP, deterministic
social intelligence pipeline with:
- Sentiment analysis (positive/negative/neutral + aggregate score)
- Sentiment-over-time tracking (history inside topic profile)
- Trend detection (emerging/growing/declining/viral spikes via rolling windows)
- Manipulation/bot detection (heuristic burstiness + repetition + coordination)
- Community analysis (volume, engagement growth, participation, sentiment evolution)
- MemoryAI integration: stores per-topic intelligence into `knowledge_records`

No network collectors are implemented. `collect_signals()` returns deterministic
sample events or uses seed payloads when provided.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ----------------------------- utilities ---------------------------------


def _utc_ts(x: Any) -> float:
    """Convert many timestamp shapes into epoch seconds."""
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return 0.0

    # Handle ISO8601
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).timestamp()
    except Exception:
        pass

    # Handle numeric string
    try:
        return float(s)
    except Exception:
        return 0.0


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _sigmoid(x: float) -> float:
    # stable-ish logistic
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return [t for t in re.split(r"\W+", text) if t]


def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


# ----------------------------- sentiment ----------------------------------


_POS = {
    "moon",
    "bull",
    "great",
    "good",
    "love",
    "win",
    "wins",
    "massive",
    "amazing",
    "bullish",
    "support",
    "breakout",
    "recover",
    "surge",
    "safest",
    "best",
    "legit",
}

_NEG = {
    "scam",
    "dump",
    "rug",
    "fraud",
    "fake",
    "bad",
    "hate",
    "lose",
    "losing",
    "terrible",
    "worst",
    "bearish",
    "panic",
    "crash",
    "manipulated",
    "hacked",
    "fud",
}

_NEGATIONS = {"not", "no", "never", "dont", "don't", "isnt", "isn't"}


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Return per-text sentiment classification and score.

    Returns:
      {
        "positive": bool,
        "negative": bool,
        "neutral": bool,
        "sentiment_score": float in [-1, 1]
      }
    """
    tokens = _tokenize(text)
    if not tokens:
        return {
            "positive": False,
            "negative": False,
            "neutral": True,
            "sentiment_score": 0.0,
        }

    score = 0.0
    pos_hits = 0
    neg_hits = 0

    # Simple negation handling: if negation appears within window before token, flip polarity.
    window = 3
    for i, tok in enumerate(tokens):
        if tok in _POS or tok in _NEG:
            is_negated = any(t in _NEGATIONS for t in tokens[max(0, i - window) : i])
            val = 1.0 if tok in _POS else -1.0
            if is_negated:
                val *= -1.0
            score += val
            if val > 0:
                pos_hits += 1
            else:
                neg_hits += 1

    # Normalize
    # score range roughly [-len, len] => map to [-1,1]
    denom = max(1.0, float(pos_hits + neg_hits))
    score_norm = score / denom
    score_norm = max(-1.0, min(1.0, score_norm))

    if pos_hits == 0 and neg_hits == 0:
        return {
            "positive": False,
            "negative": False,
            "neutral": True,
            "sentiment_score": 0.0,
        }

    if score_norm > 0.15:
        return {
            "positive": True,
            "negative": False,
            "neutral": False,
            "sentiment_score": float(score_norm),
        }
    if score_norm < -0.15:
        return {
            "positive": False,
            "negative": True,
            "neutral": False,
            "sentiment_score": float(score_norm),
        }

    return {
        "positive": False,
        "negative": False,
        "neutral": True,
        "sentiment_score": float(score_norm),
    }


def sentiment_score_to_bucket(score: float) -> str:
    if score > 0.15:
        return "positive"
    if score < -0.15:
        return "negative"
    return "neutral"


# ----------------------------- trend detection ------------------------------


def _rolling_windows(events: List[Dict[str, Any]], window_seconds: float) -> List[Tuple[float, float, List[Dict[str, Any]]]]:
    """Create time windows over events sorted by time.

    Returns list of (start, end, events_in_window)
    """
    if not events:
        return []

    times = sorted(_utc_ts(e.get("timestamp")) for e in events)
    start_t = min(times)
    end_t = max(times)
    if window_seconds <= 0:
        window_seconds = 3600

    windows = []
    t = start_t
    while t <= end_t:
        w_start = t
        w_end = t + window_seconds
        in_win = [e for e in events if w_start <= _utc_ts(e.get("timestamp")) < w_end]
        windows.append((w_start, w_end, in_win))
        t = w_end
    return windows


def detect_topic_trend(topic: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute trend score + components.

    Heuristic:
    - Volume spike based on newest vs older rolling window
    - Viral burst: high event velocity
    - Sentiment acceleration: recent sentiment magnitude moving away from neutral

    Returns:
      {
        "trend_score": float [0,1],
        "components": {...},
        "trend_state": "emerging"|"growing"|"declining"|"viral"|"stable"
      }
    """
    if not events:
        return {
            "trend_score": 0.0,
            "components": {},
            "trend_state": "stable",
        }

    events_sorted = sorted(events, key=lambda e: _utc_ts(e.get("timestamp")))
    newest = events_sorted[-1]
    newest_t = _utc_ts(newest.get("timestamp"))
    old_threshold = newest_t - 24 * 3600
    older = [e for e in events_sorted if _utc_ts(e.get("timestamp")) < old_threshold]
    recent = [e for e in events_sorted if _utc_ts(e.get("timestamp")) >= old_threshold]

    recent_sent = [analyze_sentiment(e.get("text", "")).get("sentiment_score", 0.0) for e in recent]
    recent_volume = len(recent)
    older_volume = len(older) if older else 0

    # Rolling window spikes (use smaller window for velocity)
    window_seconds = 6 * 3600
    wins = _rolling_windows(events_sorted, window_seconds)
    win_counts = [len(w[2]) for w in wins]

    newest_win_count = win_counts[-1] if win_counts else recent_volume
    prev_win_count = win_counts[-2] if len(win_counts) >= 2 else (older_volume / max(1, (24 * 3600) // window_seconds))

    volume_ratio = (newest_win_count + 1.0) / (prev_win_count + 1.0)
    volume_spike = _clamp01((volume_ratio - 1.0) / 4.0)  # >1 => spike

    # Velocity proxy: events in last 6h
    velocity_window = 6 * 3600
    recent_velocity = len([e for e in events_sorted if newest_t - _utc_ts(e.get("timestamp")) <= velocity_window])
    velocity_score = _clamp01(recent_velocity / 30.0)

    # Sentiment acceleration: absolute sentiment away from neutral in recent
    if recent_sent:
        recent_magnitude = sum(abs(s) for s in recent_sent) / len(recent_sent)
    else:
        recent_magnitude = 0.0

    # Combine
    trend_raw = 0.45 * volume_spike + 0.35 * velocity_score + 0.20 * _clamp01(recent_magnitude)

    # State label
    if volume_spike > 0.65 and velocity_score > 0.55:
        state = "viral"
    elif recent_volume > older_volume * 1.3 and volume_spike > 0.25:
        state = "growing"
    elif recent_volume < older_volume * 0.8:
        state = "declining"
    else:
        state = "emerging" if volume_spike > 0.15 else "stable"

    return {
        "trend_score": float(_clamp01(trend_raw)),
        "components": {
            "recent_volume": recent_volume,
            "older_volume": older_volume,
            "volume_spike": float(volume_spike),
            "velocity_score": float(velocity_score),
            "recent_sentiment_magnitude": float(recent_magnitude),
            "newest_win_count": newest_win_count,
            "prev_win_count": prev_win_count,
        },
        "trend_state": state,
        "topic": topic,
    }


# ----------------------------- manipulation detection ----------------------


def detect_bot_activity(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Heuristic bot/spam detection.

    Uses:
    - Burstiness per author (many posts close in time)
    - Repetitive text fingerprints per author/topic
    - Duplicate engagement patterns

    Returns:
      {
        "bot_score": [0,1],
        "evidence": [...],
        "suspicious_authors": [...]
      }
    """
    if not events:
        return {"bot_score": 0.0, "evidence": [], "suspicious_authors": []}

    events_sorted = sorted(events, key=lambda e: _utc_ts(e.get("timestamp")))

    by_author: Dict[str, List[Dict[str, Any]]] = {}
    for e in events_sorted:
        author = str(e.get("author") or "")
        by_author.setdefault(author, []).append(e)

    suspicious_authors = []
    evidence = []
    bot_scores = []

    for author, evs in by_author.items():
        if not author:
            continue
        times = [_utc_ts(e.get("timestamp")) for e in evs]
        times.sort()

        # Burstiness: how many within smallest gaps
        gaps = [times[i + 1] - times[i] for i in range(len(times) - 1) if (times[i + 1] - times[i]) > 0]
        min_gap = min(gaps) if gaps else None

        # Repetition: identical text hashes
        hashes = [_hash_text(e.get("text", "")) for e in evs]
        uniq = len(set(hashes))
        rep_ratio = 1.0 - (uniq / max(1, len(hashes)))  # 0 => all unique, 1 => all same

        author_count = len(evs)
        burst_score = 0.0
        if min_gap is not None:
            # <60s => very bursty
            burst_score = _clamp01((60.0 - min_gap) / 600.0)

        # Frequency normalization
        freq_score = _clamp01(author_count / 25.0)

        # Engagement similarity proxy
        engagements = [_safe_float(e.get("engagement", 0.0)) for e in evs]
        avg_eng = sum(engagements) / max(1, len(engagements))
        # if avg is low doesn't matter; instead if std is tiny => templated
        mean = avg_eng
        var = sum((x - mean) ** 2 for x in engagements) / max(1, len(engagements))
        std = math.sqrt(var)
        templated_score = _clamp01(1.0 - (std / (mean + 1e-6)))

        composite = 0.45 * burst_score + 0.35 * rep_ratio + 0.20 * (0.6 * freq_score + 0.4 * templated_score)
        bot_scores.append(composite)

        if composite > 0.55:
            suspicious_authors.append(author)
            evidence.append({
                "author": author,
                "bot_composite": float(composite),
                "min_gap_seconds": min_gap,
                "repetition_ratio": float(rep_ratio),
                "freq_score": float(freq_score),
                "templated_score": float(templated_score),
            })

    overall = max(bot_scores) if bot_scores else 0.0
    overall = _clamp01(overall)
    return {
        "bot_score": float(overall),
        "evidence": evidence,
        "suspicious_authors": suspicious_authors[:10],
    }


def detect_coordinated_campaign(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect coordinated behavior across authors.

    Heuristic:
    - Many distinct authors share high similarity to same template fingerprint
    - Many authors post within a short time window

    Returns:
      {"coordination_score": [0,1], "evidence": [...]} 
    """
    if not events:
        return {"coordination_score": 0.0, "evidence": []}

    events_sorted = sorted(events, key=lambda e: _utc_ts(e.get("timestamp")))
    newest_t = _utc_ts(events_sorted[-1].get("timestamp"))

    short_window = 4 * 3600
    window_events = [e for e in events_sorted if newest_t - _utc_ts(e.get("timestamp")) <= short_window]

    authors = [str(e.get("author") or "") for e in window_events]
    authors = [a for a in authors if a]

    # Template fingerprint distribution
    by_fp: Dict[str, List[Dict[str, Any]]] = {}
    for e in window_events:
        fp = _hash_text(e.get("text", ""))
        by_fp.setdefault(fp, []).append(e)

    top_fp = None
    top_count = 0
    top_authors = []
    for fp, evs in sorted(by_fp.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]:
        if len(evs) > top_count:
            top_fp = fp
            top_count = len(evs)
            top_authors = list({str(e.get("author") or "") for e in evs if str(e.get("author") or "")})

    distinct_authors = len(set(a for a in authors))
    fp_concentration = _clamp01((top_count + 1.0) / (len(window_events) + 1.0))
    author_breadth = _clamp01(distinct_authors / 25.0)

    # Coordination: high concentration + broad participation
    coordination_score = _clamp01(0.65 * fp_concentration + 0.35 * author_breadth)

    evidence = []
    if coordination_score > 0.4 and top_fp is not None:
        evidence.append({
            "time_window_seconds": short_window,
            "top_fingerprint": top_fp,
            "top_fingerprint_posts": top_count,
            "distinct_authors_in_window": distinct_authors,
            "distinct_authors_in_top_fingerprint": len(top_authors),
            "fp_concentration": float(fp_concentration),
            "author_breadth": float(author_breadth),
        })

    return {"coordination_score": float(coordination_score), "evidence": evidence}



def detect_manipulation(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not events:
        return {"manipulation_score": 0.0, "bot_score": 0.0, "coordination_score": 0.0, "evidence": {}}

    bot = detect_bot_activity(events)
    coord = detect_coordinated_campaign(events)

    # Additional spam activity: duplicate URLs if present
    urls = [str(e.get("url") or "").strip() for e in events if str(e.get("url") or "").strip()]
    dup_ratio = 0.0
    if urls:
        dup_ratio = 1.0 - (len(set(urls)) / max(1, len(urls)))
        dup_ratio = _clamp01(dup_ratio)

    spam_score = dup_ratio

    manipulation_raw = 0.45 * bot.get("bot_score", 0.0) + 0.40 * coord.get("coordination_score", 0.0) + 0.15 * spam_score
    manipulation_score = _clamp01(manipulation_raw)

    evidence = {
        "bot": bot,
        "coordination": coord,
        "spam_url_duplication_score": float(spam_score),
    }

    return {
        "manipulation_score": float(manipulation_score),
        "bot_score": float(bot.get("bot_score", 0.0)),
        "coordination_score": float(coord.get("coordination_score", 0.0)),
        "evidence": evidence,
    }


# ----------------------------- community analysis --------------------------


def analyze_community(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-topic community metrics.

    Returns:
      {
        "discussion_volume": int,
        "engagement_growth": float [0,1],
        "sentiment_evolution": {...},
        "community_participation": {...}
      }
    """
    if not events:
        return {
            "discussion_volume": 0,
            "engagement_growth": 0.0,
            "sentiment_evolution": {},
            "community_participation": {},
        }

    events_sorted = sorted(events, key=lambda e: _utc_ts(e.get("timestamp")))
    newest_t = _utc_ts(events_sorted[-1].get("timestamp"))

    # Engagement extraction
    def engagement(e: Dict[str, Any]) -> float:
        return _safe_float(e.get("engagement"), 0.0)

    recent_window = 24 * 3600
    recent = [e for e in events_sorted if newest_t - _utc_ts(e.get("timestamp")) <= recent_window]
    older = [e for e in events_sorted if newest_t - _utc_ts(e.get("timestamp")) > recent_window]

    recent_eng_avg = (sum(engagement(e) for e in recent) / max(1, len(recent))) if recent else 0.0
    older_eng_avg = (sum(engagement(e) for e in older) / max(1, len(older))) if older else 0.0

    growth_raw = 0.0
    if older_eng_avg > 0:
        growth_raw = (recent_eng_avg - older_eng_avg) / older_eng_avg
    else:
        growth_raw = 1.0 if recent_eng_avg > 0 else 0.0

    engagement_growth = _clamp01((growth_raw + 0.2) / 1.5)  # shift/scale

    # Participation: unique authors and distribution
    authors = [str(e.get("author") or "") for e in events_sorted if str(e.get("author") or "")]
    uniq_authors = len(set(authors))

    # Thread propagation proxy
    thread_ids = [str(e.get("thread_id") or "") for e in events_sorted if str(e.get("thread_id") or "")]
    uniq_threads = len(set(thread_ids))

    participation = {
        "unique_authors": uniq_authors,
        "unique_threads": uniq_threads,
        "participation_density": _clamp01(uniq_authors / max(1, len(events_sorted))),
    }

    # Sentiment evolution: average sentiment score in first half vs second half
    mid = len(events_sorted) // 2
    first = events_sorted[:mid] if mid > 0 else events_sorted
    second = events_sorted[mid:] if mid < len(events_sorted) else events_sorted

    def avg_sent(es: List[Dict[str, Any]]) -> float:
        if not es:
            return 0.0
        vals = [analyze_sentiment(e.get("text", "")).get("sentiment_score", 0.0) for e in es]
        return sum(vals) / max(1, len(vals))

    s1 = avg_sent(first)
    s2 = avg_sent(second)

    # Evolution: how far sentiment shifts (toward positivity/negativity)
    delta = s2 - s1
    sentiment_evolution = {
        "sentiment_first_half": float(s1),
        "sentiment_second_half": float(s2),
        "delta": float(delta),
        "trend_direction": "up" if delta > 0 else "down" if delta < 0 else "flat",
    }

    return {
        "discussion_volume": int(len(events_sorted)),
        "engagement_growth": float(engagement_growth),
        "sentiment_evolution": sentiment_evolution,
        "community_participation": participation,
    }


# ----------------------------- pipeline assembly ---------------------------


def _group_by_topic(signals: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for s in signals:
        topic = str(s.get("topic") or "").strip() or "general"
        grouped.setdefault(topic, []).append(s)
    return grouped


def build_topic_profile(topic: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    # sentiment per event + history
    sentiment_history = []
    sentiment_scores = []
    bucket_counts = {"positive": 0, "negative": 0, "neutral": 0}

    for e in sorted(events, key=lambda x: _utc_ts(x.get("timestamp"))):
        text = str(e.get("text") or "")
        s = analyze_sentiment(text)
        score = float(s.get("sentiment_score", 0.0))
        bucket = sentiment_score_to_bucket(score)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        sentiment_history.append({
            "timestamp": _utc_ts(e.get("timestamp")),
            "author": str(e.get("author") or ""),
            "sentiment_bucket": bucket,
            "sentiment_score": score,
        })
        sentiment_scores.append(score)

    # sentiment_score in output should be in [0,1] not [-1,1]
    if sentiment_scores:
        avg = sum(sentiment_scores) / max(1, len(sentiment_scores))
    else:
        avg = 0.0

    sentiment_score_01 = _clamp01((avg + 1.0) / 2.0)

    trend = detect_topic_trend(topic, events)
    manip = detect_manipulation(events)
    community = analyze_community(events)

    # confidence: higher if more evidence + lower manipulation + stronger trend
    volume = len(events)
    evidence_score = _clamp01(volume / 50.0)
    manipulation_penalty = _clamp01(manip.get("manipulation_score", 0.0))

    confidence = _clamp01(
        0.30 * evidence_score + 0.35 * trend.get("trend_score", 0.0) + 0.20 * community.get("engagement_growth", 0.0) + 0.15 * (1.0 - manipulation_penalty)
    )

    return {
        "topic": topic,
        "discussion_volume": int(volume),
        "sentiment": {
            "avg_sentiment_score_01": float(sentiment_score_01),
            "bucket_counts": bucket_counts,
            "history": sentiment_history[-50:],  # keep bounded
        },
        "trend": {
            "trend_score": float(trend.get("trend_score", 0.0)),
            "trend_state": trend.get("trend_state", "stable"),
            "components": trend.get("components", {}),
        },
        "manipulation": {
            "manipulation_score": float(manip.get("manipulation_score", 0.0)),
            "bot_score": float(manip.get("bot_score", 0.0)),
            "coordination_score": float(manip.get("coordination_score", 0.0)),
            "evidence": manip.get("evidence", {}),
        },
        "community": community,
        "confidence_score": float(confidence),
        "_debug": {
            "evidence_score": float(evidence_score),
            "manipulation_penalty": float(manipulation_penalty),
        },
    }


OUTPUT_TEMPLATE = {
    "topic": "",
    "trend_score": 0,
    "sentiment_score": 0,
    "manipulation_score": 0,
    "confidence_score": 0,
    "summary": "",
}


def build_topic_output(topic_profile: Dict[str, Any]) -> Dict[str, Any]:
    topic = str(topic_profile.get("topic", ""))
    trend_score = float(topic_profile.get("trend", {}).get("trend_score", 0.0))
    sentiment_score = float(topic_profile.get("sentiment", {}).get("avg_sentiment_score_01", 0.0))
    manipulation_score = float(topic_profile.get("manipulation", {}).get("manipulation_score", 0.0))
    confidence_score = float(topic_profile.get("confidence_score", 0.0))

    # Compose compact deterministic summary
    bucket_counts = topic_profile.get("sentiment", {}).get("bucket_counts", {})
    community = topic_profile.get("community", {})
    community_part = community.get("community_participation", {})
    trend_state = topic_profile.get("trend", {}).get("trend_state", "stable")

    summary_obj = {
        "trend_state": trend_state,
        "discussion_volume": topic_profile.get("discussion_volume", 0),
        "sentiment": {
            "avg_sentiment_score_01": sentiment_score,
            "bucket_counts": bucket_counts,
        },
        "community": {
            "engagement_growth": float(community.get("engagement_growth", 0.0)),
            "unique_authors": community_part.get("unique_authors", 0),
            "unique_threads": community_part.get("unique_threads", 0),
            "sentiment_evolution": topic_profile.get("community", {}).get("sentiment_evolution", {}),
        },
        "manipulation": {
            "bot_score": float(topic_profile.get("manipulation", {}).get("bot_score", 0.0)),
            "coordination_score": float(topic_profile.get("manipulation", {}).get("coordination_score", 0.0)),
        },
    }

    out = dict(OUTPUT_TEMPLATE)
    out.update(
        {
            "topic": topic,
            "trend_score": trend_score,
            "sentiment_score": sentiment_score,
            "manipulation_score": manipulation_score,
            "confidence_score": confidence_score,
            "summary": json.dumps(summary_obj, ensure_ascii=False, separators=(",", ":")),
        }
    )

    # Enforce output keys
    return {k: out[k] for k in OUTPUT_TEMPLATE.keys()}


# ----------------------------- collectors (MVP stub) ------------------------


def collect_signals(seed_items: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Collect raw social signals.

    MVP-safe behavior:
    - if `seed_items` is provided, return them normalized.
    - else, attempt a Reddit-backed fetch if credentials are available.
    - if credentials/network are missing, return the deterministic sample
      dataset (so the rest of the pipeline keeps working).
    """


    Expected seed format (best-effort):
      {
        "topic": "...",
        "platform": "x"|"reddit"|..., 
        "source": "x"|..., 
        "text": "...",
        "author": "...",
        "timestamp": <epoch seconds or ISO8601>,
        "engagement": <float>,
        "thread_id": "..." (optional),
        "url": "..." (optional)
      }
    """
    if seed_items is not None:
        return [_normalize_signal(s) for s in seed_items]

    # Deterministic sample events
    base_t = 1710000000  # fixed epoch seconds

    sample = [
        {
            "topic": "BTC",
            "platform": "x",
            "source": "x",
            "text": "BTC is massive and bullish, we are going to the moon!",
            "author": "alice",
            "timestamp": base_t + 3600 * 1,
            "engagement": 120,
            "thread_id": "t1",
        },
        {
            "topic": "BTC",
            "platform": "x",
            "source": "x",
            "text": "BTC bullish breakout soon. legit win, support!",
            "author": "bob",
            "timestamp": base_t + 3600 * 2,
            "engagement": 90,
            "thread_id": "t1",
        },
        {
            "topic": "BTC",
            "platform": "reddit",
            "source": "reddit",
            "text": "Love the surge, best recovery ever. Great stuff!",
            "author": "carol",
            "timestamp": base_t + 3600 * 22,
            "engagement": 210,
            "thread_id": "t2",
        },
        # Burst + repetition suggestive of coordination/manipulation
        {
            "topic": "MEME",
            "platform": "x",
            "source": "x",
            "text": "MEME to the moon! legit massive win win win!",
            "author": "bot1",
            "timestamp": base_t + 3600 * 30,
            "engagement": 15,
            "thread_id": "m1",
            "url": "http://example.com/pump",
        },
        {
            "topic": "MEME",
            "platform": "x",
            "source": "x",
            "text": "MEME to the moon! legit massive win win win!",
            "author": "bot2",
            "timestamp": base_t + 3600 * 30 + 60,
            "engagement": 16,
            "thread_id": "m1",
            "url": "http://example.com/pump",
        },
        {
            "topic": "MEME",
            "platform": "discord",
            "source": "discord",
            "text": "MEME to the moon! legit massive win win win!",
            "author": "bot3",
            "timestamp": base_t + 3600 * 30 + 120,
            "engagement": 14,
            "thread_id": "m1",
            "url": "http://example.com/pump",
        },
        # Negative narrative/panic
        {
            "topic": "ALT",
            "platform": "forum",
            "source": "forum",
            "text": "This is a scam, panic dump, worst fake project.",
            "author": "dave",
            "timestamp": base_t + 3600 * 10,
            "engagement": 40,
            "thread_id": "a1",
        },
        {
            "topic": "ALT",
            "platform": "forum",
            "source": "forum",
            "text": "FUD everywhere. hacked? manipulated? rug?",
            "author": "erin",
            "timestamp": base_t + 3600 * 12,
            "engagement": 65,
            "thread_id": "a1",
        },
    ]

    return [_normalize_signal(s) for s in sample]


def _normalize_signal(s: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "topic": str(s.get("topic") or "general").strip() or "general",
        "platform": str(s.get("platform") or s.get("source") or ""),
        "source": str(s.get("source") or s.get("platform") or ""),
        "text": str(s.get("text") or s.get("content") or ""),
        "author": str(s.get("author") or ""),
        "timestamp": s.get("timestamp") or s.get("time") or datetime.now(timezone.utc).isoformat(),
        "engagement": _safe_float(s.get("engagement"), 0.0) + _safe_float(s.get("likes"), 0.0) + _safe_float(s.get("replies"), 0.0) * 0.5 + _safe_float(s.get("shares"), 0.0) * 0.7,
        "thread_id": str(s.get("thread_id") or s.get("conversation_id") or ""),
        "url": str(s.get("url") or ""),
        "author_is_bot": bool(s.get("author_is_bot")) if "author_is_bot" in s else False,
        "raw": s,
    }


# ----------------------------- orchestration --------------------------------


def classify_manipulation(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """MVP manipulation classifier.

    For now, it just annotates each signal with a bot/coordination flag
    derived from the author's repetition/templatization.
    """
    # This function is not used by build_topic_profile (it recomputes from events).
    # Kept for compatibility with existing placeholder code.
    out = []
    for s in signals:
        text = str(s.get("text") or "")
        fp = _hash_text(text)
        # crude per-signal template detection
        repeated = len(text) > 0 and ("win" in text.lower() or "moon" in text.lower())
        out.append({**s, "fingerprint": fp, "suspected_spam": bool(repeated and ("http" in (s.get("url") or "")))})
    return out


def _stable_knowledge_id(topic: str, run_bucket: str) -> str:
    # stable id for per-topic profiles per time bucket (run_bucket is deterministic)
    base = f"socialintel|topic={topic}|bucket={run_bucket}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:32]


def store_to_memory(
    memory_ai: Any,
    domain: str = "social",
    seed_items: Optional[List[Dict[str, Any]]] = None,
    run_bucket: Optional[str] = None,
) -> Dict[str, Any]:
    """Store per-topic intelligence into MemoryAI.

    Creates one `knowledge_records` entry per topic per run_bucket.

    Returns:
      {"stored": int, "topics": [...], "signals_seen": int}
    """

    signals = collect_signals(seed_items=seed_items)
    grouped = _group_by_topic(signals)

    if run_bucket is None:
        # 6-hour deterministic bucket using newest signal timestamp
        newest = max(_utc_ts(s.get("timestamp")) for s in signals) if signals else 0.0
        bucket_start = int(newest // (6 * 3600)) * (6 * 3600)
        run_bucket = str(bucket_start)

    stored = 0
    topics_out = []

    for topic, events in grouped.items():
        profile = build_topic_profile(topic, events)
        output = build_topic_output(profile)

        knowledge_id = _stable_knowledge_id(topic=topic, run_bucket=run_bucket)
        summary_text = output["summary"]

        relationships = []
        # Optional: store trend/manipulation evidence as relationship metadata (stub)
        relationships.append({
            "type": "topic_profile",
            "run_bucket": run_bucket,
            "evidence": {
                "trend_state": profile.get("trend", {}).get("trend_state"),
            },
        })

        if hasattr(memory_ai, "store_knowledge_record"):
            memory_ai.store_knowledge_record(
                knowledge_id=knowledge_id,
                domain=domain,
                source="SocialIntelService",
                title=f"social_topic:{topic}",
                author="",
                category="social_topic_profile",
                symbol=topic,
                strategy_type="",
                market_regime="",
                importance_score=float(output["trend_score"]),
                embedding=[],
                summary=summary_text,
                relationships=relationships,
            )
            stored += 1

        topics_out.append(output)

    return {"stored": stored, "topics": [t["topic"] for t in topics_out], "signals_seen": len(signals), "run_bucket": run_bucket}

