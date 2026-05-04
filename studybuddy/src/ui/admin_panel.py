"""
studybuddy/src/ui/admin_panel.py

Admin dashboard for DataPilot — professor/admin view only.
Called from app.py when page == "Admin Panel" and is_admin is True.

Public API:
  render() -> None
"""

import sys
from pathlib import Path
_SRC_DIR = Path(__file__).resolve().parent.parent   # studybuddy/src/
_PKG_DIR = _SRC_DIR.parent                          # studybuddy/
for _p in [str(_SRC_DIR), str(_PKG_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datetime import datetime, timezone, timedelta

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.db.mongo_client import (
    get_users_collection,
    get_attempts_collection,
    get_weak_interactions_collection,
)


# ── Cached data-fetch helpers ─────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_overview() -> dict:
    try:
        users    = get_users_collection()
        attempts = get_attempts_collection()

        total_students = users.count_documents({"role": {"$ne": "admin"}})
        total_attempts = attempts.count_documents({})

        # Average score
        pipe_avg = [{"$group": {"_id": None, "avg": {"$avg": "$percentage"}}}]
        avg_rows = list(attempts.aggregate(pipe_avg))
        avg_score = round(avg_rows[0]["avg"], 1) if avg_rows else 0.0

        # Active today (UTC date)
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        active_today = attempts.count_documents({"timestamp": {"$gte": today_start}})

        # Attempts per day — last 30 days
        thirty_ago = datetime.now(timezone.utc) - timedelta(days=30)
        pipe_daily = [
            {"$match": {"timestamp": {"$gte": thirty_ago}}},
            {"$group": {
                "_id": {
                    "y": {"$year":  "$timestamp"},
                    "m": {"$month": "$timestamp"},
                    "d": {"$dayOfMonth": "$timestamp"},
                },
                "count": {"$sum": 1},
            }},
            {"$sort": {"_id.y": 1, "_id.m": 1, "_id.d": 1}},
        ]
        daily_rows = list(attempts.aggregate(pipe_daily))
        daily_dates  = []
        daily_counts = []
        for row in daily_rows:
            d = row["_id"]
            daily_dates.append(f"{d['y']:04d}-{d['m']:02d}-{d['d']:02d}")
            daily_counts.append(row["count"])

        return {
            "total_students": total_students,
            "total_attempts": total_attempts,
            "avg_score":      avg_score,
            "active_today":   active_today,
            "daily_dates":    daily_dates,
            "daily_counts":   daily_counts,
        }
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_performance() -> dict:
    try:
        attempts = get_attempts_collection()

        all_scores = [
            doc["percentage"]
            for doc in attempts.find({}, {"percentage": 1, "_id": 0})
            if doc.get("percentage") is not None
        ]

        pipe_per_student = [
            {"$group": {
                "_id":       "$username",
                "avg_score": {"$avg": "$percentage"},
                "attempts":  {"$sum": 1},
            }},
            {"$sort": {"avg_score": -1}},
            {"$limit": 20},
        ]
        student_rows = list(attempts.aggregate(pipe_per_student))

        pipe_per_diff = [
            {"$group": {
                "_id":       {"$ifNull": ["$difficulty", "Medium"]},
                "avg_score": {"$avg": "$percentage"},
                "attempts":  {"$sum": 1},
            }},
        ]
        diff_rows = list(attempts.aggregate(pipe_per_diff))

        return {
            "all_scores":    all_scores,
            "student_rows":  student_rows,
            "diff_rows":     diff_rows,
        }
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_knowledge_gaps() -> dict:
    try:
        weak = get_weak_interactions_collection()

        pipe_topics = [
            {"$group": {
                "_id":    "$topic",
                "count":  {"$sum": 1},
                "source": {"$first": "$source_file"},
                "students": {"$addToSet": "$username"},
            }},
            {"$sort": {"count": -1}},
            {"$limit": 10},
        ]
        topic_rows = list(weak.aggregate(pipe_topics))

        pipe_sources = [
            {"$group": {"_id": "$source_file", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10},
        ]
        source_rows = list(weak.aggregate(pipe_sources))

        # Full table — top 30 topics with student count
        pipe_table = [
            {"$group": {
                "_id":      "$topic",
                "failures": {"$sum": 1},
                "source":   {"$first": "$source_file"},
                "students": {"$addToSet": "$username"},
            }},
            {"$sort": {"failures": -1}},
            {"$limit": 30},
        ]
        table_rows = list(weak.aggregate(pipe_table))

        return {
            "topic_rows":  topic_rows,
            "source_rows": source_rows,
            "table_rows":  table_rows,
        }
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_students() -> dict:
    try:
        attempts = get_attempts_collection()
        weak     = get_weak_interactions_collection()

        pipe_students = [
            {"$group": {
                "_id":         "$username",
                "total_quizzes": {"$sum": 1},
                "avg_score":   {"$avg": "$percentage"},
                "last_active": {"$max": "$timestamp"},
            }},
            {"$sort": {"username": 1}},
        ]
        student_rows = list(attempts.aggregate(pipe_students))

        # Weak-topic count per student
        pipe_weak = [
            {"$group": {
                "_id":        "$username",
                "weak_count": {"$sum": 1},
                "top_topic":  {"$first": "$topic"},
            }},
        ]
        weak_by_user = {
            r["_id"]: {"weak_count": r["weak_count"], "top_topic": r["top_topic"]}
            for r in weak.aggregate(pipe_weak)
        }

        rows = []
        for r in student_rows:
            uname    = r["_id"]
            weak_inf = weak_by_user.get(uname, {"weak_count": 0, "top_topic": "—"})
            last_dt  = r.get("last_active")
            last_str = last_dt.strftime("%Y-%m-%d") if isinstance(last_dt, datetime) else "—"
            rows.append({
                "Username":       uname,
                "Quizzes Taken":  r["total_quizzes"],
                "Avg Score (%)":  round(r["avg_score"], 1),
                "Weak Topics":    weak_inf["weak_count"],
                "Top Weak Topic": weak_inf["top_topic"] or "—",
                "Last Active":    last_str,
            })

        return {"rows": rows}
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_student_history(username: str) -> list[dict]:
    try:
        attempts = get_attempts_collection()
        docs = (
            attempts
            .find({"username": username.strip().lower()})
            .sort("timestamp", 1)
        )
        result = []
        for i, doc in enumerate(docs, start=1):
            ts = doc.get("timestamp")
            result.append({
                "Session":     i,
                "Topic":       doc.get("topic", ""),
                "Difficulty":  doc.get("difficulty", "Medium"),
                "Score (%)":   doc.get("percentage", 0.0),
                "Correct":     doc.get("score", 0),
                "Total Qs":    doc.get("total", 0),
                "Date":        ts.strftime("%Y-%m-%d %H:%M") if isinstance(ts, datetime) else "—",
            })
        return result
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_at_risk() -> dict:
    try:
        attempts = get_attempts_collection()
        weak     = get_weak_interactions_collection()

        now = datetime.now(timezone.utc)

        pipe_students = [
            {"$group": {
                "_id":         "$username",
                "avg_score":   {"$avg": "$percentage"},
                "last_active": {"$max": "$timestamp"},
            }},
        ]
        student_rows = list(attempts.aggregate(pipe_students))

        pipe_weak = [
            {"$group": {"_id": "$username", "weak_count": {"$sum": 1}}},
        ]
        weak_counts = {r["_id"]: r["weak_count"] for r in weak.aggregate(pipe_weak)}

        rows = []
        for r in student_rows:
            uname      = r["_id"]
            avg        = r.get("avg_score") or 0.0
            last_dt    = r.get("last_active")
            days_inactive = (
                (now - last_dt.replace(tzinfo=timezone.utc)
                 if last_dt.tzinfo is None else now - last_dt).days
                if isinstance(last_dt, datetime) else 30
            )
            wc    = weak_counts.get(uname, 0)
            risk  = min(100, (100 - avg) * 0.5 + wc * 5 + days_inactive * 2)
            level = "High" if risk > 60 else ("Medium" if risk >= 30 else "Low")
            rows.append({
                "username":      uname,
                "risk_score":    round(risk, 1),
                "risk_level":    level,
                "avg_score":     round(avg, 1),
                "weak_count":    wc,
                "days_inactive": days_inactive,
            })

        rows.sort(key=lambda x: x["risk_score"], reverse=True)
        return {"rows": rows}
    except Exception as e:
        return {"error": str(e)}


# ── Tab renderers ─────────────────────────────────────────────────────────────

def _tab_overview():
    data = _fetch_overview()
    if "error" in data:
        st.error(f"Could not load overview data: {data['error']}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Total Students",    data["total_students"])
    c2.metric("📝 Total Quiz Attempts", data["total_attempts"])
    c3.metric("🎯 Avg Score",          f"{data['avg_score']}%")
    c4.metric("🟢 Active Today",       data["active_today"])

    st.markdown("---")
    st.subheader("Quiz Attempts — Last 30 Days")

    if not data["daily_dates"]:
        st.info("No quiz attempts in the last 30 days.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["daily_dates"],
        y=data["daily_counts"],
        mode="lines+markers",
        line=dict(color="#2ecc71", width=2),
        marker=dict(size=6),
        fill="tozeroy",
        fillcolor="rgba(46,204,113,0.15)",
    ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Attempts",
        margin=dict(l=20, r=20, t=20, b=20),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


def _tab_performance():
    data = _fetch_performance()
    if "error" in data:
        st.error(f"Could not load performance data: {data['error']}")
        return

    if not data["all_scores"]:
        st.info("No quiz attempts yet — no data to display.")
        return

    # Score distribution histogram
    st.subheader("Score Distribution")
    fig_hist = go.Figure(go.Histogram(
        x=data["all_scores"],
        xbins=dict(start=0, end=100, size=10),
        marker_color="#3498db",
        opacity=0.8,
    ))
    fig_hist.update_layout(
        xaxis_title="Score (%)",
        yaxis_title="Number of Attempts",
        margin=dict(l=20, r=20, t=20, b=20),
        height=280,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    col_l, col_r = st.columns(2)

    # Average score per student (top 20)
    with col_l:
        st.subheader("Top 20 Students by Avg Score")
        if data["student_rows"]:
            names  = [r["_id"]     for r in data["student_rows"]]
            scores = [round(r["avg_score"], 1) for r in data["student_rows"]]
            fig_st = go.Figure(go.Bar(
                x=scores, y=names, orientation="h",
                marker_color="#2ecc71", opacity=0.85,
            ))
            fig_st.update_layout(
                xaxis_title="Avg Score (%)",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=20, r=20, t=20, b=20),
                height=max(250, len(names) * 22),
            )
            st.plotly_chart(fig_st, use_container_width=True)
        else:
            st.info("No data yet.")

    # Average score per difficulty
    with col_r:
        st.subheader("Avg Score by Difficulty")
        if data["diff_rows"]:
            diffs  = [r["_id"] or "Medium" for r in data["diff_rows"]]
            dscores = [round(r["avg_score"], 1) for r in data["diff_rows"]]
            color_map = {"Easy": "#2ecc71", "Medium": "#f39c12", "Hard": "#e74c3c"}
            colors = [color_map.get(d, "#95a5a6") for d in diffs]
            fig_df = go.Figure(go.Bar(
                x=diffs, y=dscores,
                marker_color=colors, opacity=0.85,
            ))
            fig_df.update_layout(
                yaxis_title="Avg Score (%)",
                margin=dict(l=20, r=20, t=20, b=20),
                height=280,
            )
            st.plotly_chart(fig_df, use_container_width=True)
        else:
            st.info("No data yet.")


def _tab_knowledge_gaps():
    data = _fetch_knowledge_gaps()
    if "error" in data:
        st.error(f"Could not load knowledge gap data: {data['error']}")
        return

    if not data["topic_rows"] and not data["source_rows"]:
        st.info("No weak interaction data yet. This populates as students take quizzes.")
        return

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Top 10 Most Failed Topics")
        if data["topic_rows"]:
            topics = [r["_id"] or "Unknown" for r in data["topic_rows"]]
            counts = [r["count"] for r in data["topic_rows"]]
            fig = go.Figure(go.Bar(
                x=counts, y=topics, orientation="h",
                marker_color="#e74c3c", opacity=0.8,
            ))
            fig.update_layout(
                xaxis_title="Failure Count",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=20, r=20, t=20, b=20),
                height=max(250, len(topics) * 25),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No failed topics recorded yet.")

    with col_r:
        st.subheader("Failures by Source PDF")
        if data["source_rows"]:
            sources = [r["_id"] or "Unknown" for r in data["source_rows"]]
            scounts = [r["count"] for r in data["source_rows"]]
            fig_pie = go.Figure(go.Pie(
                labels=sources, values=scounts,
                hole=0.35,
                textinfo="label+percent",
            ))
            fig_pie.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=320,
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No data yet.")

    st.markdown("---")
    st.info("💡 These are the topics your professor should review in class.")

    if data["table_rows"]:
        st.subheader("Failure Detail Table")
        table_data = []
        for r in data["table_rows"]:
            table_data.append({
                "Topic":            r["_id"] or "Unknown",
                "Failure Count":    r["failures"],
                "Students Affected": len(r.get("students", [])),
                "Source PDF":        r.get("source", "—") or "—",
            })
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)


def _tab_student_details():
    data = _fetch_students()
    if "error" in data:
        st.error(f"Could not load student data: {data['error']}")
        return

    if not data["rows"]:
        st.info("No student data yet.")
        return

    st.subheader("All Students")
    df = pd.DataFrame(data["rows"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Student Quiz History")
    usernames = [r["Username"] for r in data["rows"]]
    selected  = st.selectbox("Select a student to view their full quiz history:", usernames)

    if selected:
        history = _fetch_student_history(selected)
        if not history:
            st.info(f"No quiz history found for **{selected}**.")
        else:
            hist_df = pd.DataFrame(history)
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

            # Timeline chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[r["Date"]      for r in history],
                y=[r["Score (%)"] for r in history],
                mode="lines+markers",
                line=dict(color="#3498db", width=2),
                marker=dict(size=7),
                name="Score",
            ))
            fig.add_hline(y=50, line_dash="dash", line_color="gray",
                          annotation_text="50% threshold")
            fig.update_layout(
                title=f"Score timeline — {selected}",
                xaxis_title="Date",
                yaxis_title="Score (%)",
                yaxis=dict(range=[0, 105]),
                margin=dict(l=20, r=20, t=40, b=20),
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True)


def _tab_at_risk():
    data = _fetch_at_risk()
    if "error" in data:
        st.error(f"Could not load at-risk data: {data['error']}")
        return

    if not data["rows"]:
        st.info("No student data yet.")
        return

    st.warning("⚠️ High-risk students may need additional support before the exam.")

    rows = data["rows"]
    names  = [r["username"]   for r in rows]
    scores = [r["risk_score"] for r in rows]
    colors = [
        "#e74c3c" if r["risk_score"] > 60
        else ("#f39c12" if r["risk_score"] >= 30 else "#2ecc71")
        for r in rows
    ]

    fig = go.Figure(go.Bar(
        x=names, y=scores,
        marker_color=colors, opacity=0.85,
    ))
    fig.update_layout(
        xaxis_title="Student",
        yaxis_title="Risk Score",
        yaxis=dict(range=[0, 105]),
        margin=dict(l=20, r=20, t=20, b=20),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Legend
    cl, cm, ch = st.columns(3)
    cl.markdown("🟢 **Low** risk — score < 30")
    cm.markdown("🟠 **Medium** risk — score 30–60")
    ch.markdown("🔴 **High** risk — score > 60")

    st.markdown("---")

    table_data = [{
        "Student":       r["username"],
        "Risk Score":    r["risk_score"],
        "Risk Level":    r["risk_level"],
        "Avg Score (%)": r["avg_score"],
        "Weak Topics":   r["weak_count"],
        "Days Inactive": r["days_inactive"],
    } for r in rows]
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── Public entry point ────────────────────────────────────────────────────────

def render() -> None:
    st.title("🔐 Admin Panel")
    st.caption("Professor / admin view — hidden from students")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Performance Analytics",
        "🧠 Knowledge Gaps",
        "👥 Student Details",
        "⚠️ At-Risk Students",
    ])

    with tab1:
        _tab_overview()
    with tab2:
        _tab_performance()
    with tab3:
        _tab_knowledge_gaps()
    with tab4:
        _tab_student_details()
    with tab5:
        _tab_at_risk()
