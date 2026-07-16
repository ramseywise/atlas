"""Section builders for the detailed eval stats HTML report.

Each function builds one logical block of a file section. _section() is
the compositor that assembles them into the full per-file HTML block.
"""

from __future__ import annotations

import html as _html

from evals.reports.utils._html import pct_bar_error as _pct_bar

_FT_DESCRIPTIONS: dict[str, str] = {
    "A_retrieval": "no KB answer returned (response_type = unknown) — retrieval error",
    "B_language": "response language ≠ query language",
    "C_friction": "disliked — not a grounding error; includes single dislikes, repeated queries, escalation",
    "E_grounding": "disliked + has_sources — cited but user rejected; check for hallucination",
    "unclassified": "disliked with no failure type assigned — taxonomy gap",
    "no_failure": "liked turns — no failure detected",
    "n/a": "unrated turns — no explicit signal",
}

_FT_LEGEND = (
    "<p class='muted' style='font-size:.78em;margin:2px 0 8px'>"
    "<b>Taxonomy:</b> "
    "A = no KB answer &nbsp;·&nbsp; "
    "B = language mismatch &nbsp;·&nbsp; "
    "C = friction / repetition &nbsp;·&nbsp; "
    "D = wrong escalation decision &nbsp;·&nbsp; "
    "E = cited but disliked (hallucination risk) &nbsp;·&nbsp; "
    "unclassified = disliked with no label &nbsp;·&nbsp; "
    "no_failure = liked &nbsp;·&nbsp; n/a = unrated"
    "</p>"
)

_FT_COLORS: dict[str, str] = {
    "no_failure": "#28a745",
    "n/a": "#999",
    "unclassified": "#ffc107",
}


def _summary_header(stats: dict) -> str:
    s = stats["sentiment"]
    has_src = stats.get("has_sources_rate", 0.0)
    cov = s.get("rating_coverage_rate", 0.0)
    n_unique_convs = stats.get("n_unique_convs")
    n_unique_users = stats.get("n_unique_users")
    dlr = s.get("dislike_like_ratio")

    chips = (
        f'<span class="chip liked">Liked {s["n_liked"]}</span>'
        f'<span class="chip disliked">Disliked {s["n_disliked"]}</span>'
        f'<span class="chip unrated">Unrated {s["n_unrated"]}</span>'
    )
    ratio_label = (
        f"{dlr:.1f}:1" if dlr is not None
        else ("all disliked" if s["n_disliked"] > 0 else "—")
    )

    quick_parts = []
    if n_unique_convs is not None:
        quick_parts.append(f'<span><span style="color:#666">Unique convs</span> <b>{n_unique_convs:,}</b></span>')
    if n_unique_users:
        quick_parts.append(f'<span><span style="color:#666">Unique users</span> <b>{n_unique_users:,}</b></span>')
    quick_parts.append(f'<span><span style="color:#666">Has sources</span> <b>{has_src:.1%}</b> of turns</span>')
    quick_parts.append(f'<span><span style="color:#666">Rated</span> <b>{cov:.1%}</b> of turns</span>')

    quick_strip = (
        '<div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:8px;'
        'padding:8px 10px;background:#f5f5f5;border-radius:4px;font-size:.85em">'
        + " &nbsp;·&nbsp; ".join(quick_parts) + "</div>"
    )
    return (
        f'<b>n={stats["n_total"]:,} turns</b> &nbsp; {chips}'
        f'<br/><span style="font-size:.85em;display:inline-flex;align-items:center;'
        f'gap:8px;margin-top:6px">'
        f'<span style="color:#666">Dislike:like ratio:</span>'
        f'<b style="color:#721c24">{ratio_label}</b>'
        f'</span>'
        f'{quick_strip}'
    )


def _failure_type_block(ft: dict) -> str:
    rows = ""
    for k, v in ft.items():
        ft_color = _FT_COLORS.get(k, "#e06820")
        desc = _FT_DESCRIPTIONS.get(k, "")
        desc_span = f" <span class='muted' style='font-size:.78em'>— {desc}</span>" if desc else ""
        rows += (
            f"<tr><td>{k}{desc_span}</td><td class='num'>{v['count']}</td>"
            f"<td class='num'>{v['rate']:.1%}</td>"
            f"<td>{_pct_bar(v['rate'], color=ft_color)}</td></tr>"
        )

    dominant = max(ft.items(), key=lambda x: x[1]["count"], default=None)
    tautology_note = ""
    if dominant:
        dk, dv = dominant
        if dk not in ("no_failure", "n/a") and dv["rate"] > 0.8:
            tautology_note = (
                f"<p class='muted' style='font-size:.8em;margin:2px 0 6px'>"
                f"⚠ <b>{dk}</b> dominates ({dv['rate']:.0%}) by construction — "
                f"this file was filtered on the same condition, so the label is tautological, "
                f"not a result of analysis.</p>"
            )
    return (
        "<h3>Failure types</h3>"
        f"{tautology_note}"
        "<table><tr><th>Type</th><th>Count</th><th>Rate</th><th></th></tr>"
        f"{rows}</table>"
    )


def _response_type_block(rt: dict) -> str:
    rows = "".join(
        f"<tr><td>{k}</td><td class='num'>{v['count']}</td>"
        f"<td class='num'>{v['rate']:.1%}</td></tr>"
        for k, v in rt.items()
    )
    return (
        "<h3>Response types</h3>"
        "<table><tr><th>Type</th><th>Count</th><th>Rate</th></tr>"
        f"{rows}</table>"
    )


def _category_block(cats: list, stats: dict) -> str:
    if not cats:
        return ""
    rows = "".join(
        f"<tr><td>{cat}</td><td class='num'>{cv['count']}</td>"
        f"<td class='num'>{cv['dislike_rate']:.1%}</td>"
        f"<td>{_pct_bar(cv['dislike_rate'], color='#e06820')}</td>"
        f"<td class='num'>{cv.get('sources_rate', 0):.0%}</td></tr>"
        for cat, cv in cats
    )
    n_categorized = stats["n_total"] - stats.get("no_topic_coverage", {}).get("n", 0)
    sparsity = ""
    if stats["n_total"] > 0 and n_categorized / stats["n_total"] < 0.5:
        sparsity = (
            f"<p class='muted' style='font-size:.78em;margin:2px 0 4px'>"
            f"⚠ Only {n_categorized:,}/{stats['n_total']:,} records "
            f"({n_categorized/stats['n_total']:.0%}) have a topic label — "
            f"dislike ratios are directional only.</p>"
        )
    return (
        "<h3>Topic dislike ratios (top 10)</h3>"
        f"{sparsity}"
        "<table><tr><th>Topic</th><th>n</th><th>Dislike ratio</th><th></th>"
        "<th>Has sources</th></tr>"
        f"{rows}</table>"
    )


def retrieval_proxy_block_html(stats: dict, *, variant: str = "raw") -> str:
    """Citation-proxy confusion matrix — ``raw`` (has_source) or ``overlap`` (BKH↔VA match)."""
    key = "retrieval_proxy_overlap" if variant == "overlap" else "retrieval_proxy"
    rp = stats.get(key) or {}
    if not rp:
        return ""

    s = stats.get("sentiment") or {}
    title = (
        "Retrieval proxy — BKH↔VA overlap adjusted"
        if variant == "overlap"
        else "Retrieval proxy (citation-based, pre-grader)"
    )
    y_note = (
        "ŷ = expanded slug overlap with BKH (kb_url_map) · y = VA archive liked/disliked · "
        "excludes edge_no_overlap / verify_cited_no_match rows"
        if variant == "overlap"
        else "ŷ = has_source · y = liked · unrated excluded"
    )
    proxy_tag = rp.get("_proxy", "")
    if proxy_tag:
        y_note += f" · <code>{_html.escape(proxy_tag)}</code>"

    return _retrieval_proxy_block_inner(rp, s, title=title, y_note=y_note, overlap=(variant == "overlap"))


def _retrieval_proxy_block(stats: dict) -> str:
    return retrieval_proxy_block_html(stats, variant="raw")


def _retrieval_proxy_block_inner(
    rp: dict,
    s: dict,
    *,
    title: str,
    y_note: str,
    overlap: bool = False,
) -> str:
    hit_rows = "".join(
        f"<tr><td>{s2}</td><td class='num'>{d['n']}</td>"
        f"<td class='num'>{d['hit_rate']:.1%}</td>"
        f"<td>{_pct_bar(d['hit_rate'], color='#4c9be8')}</td></tr>"
        for s2, d in rp.get("source_hit_rate_by_sentiment", {}).items()
        if d.get("hit_rate") is not None
    )

    cm = rp.get("confusion", {})
    tp, fp = cm.get("tp", 0), cm.get("fp", 0)
    fn, tn = cm.get("fn", 0), cm.get("tn", 0)
    unrated_src = cm.get("unrated_sourced", 0)
    unrated_no = cm.get("unrated_unsourced", 0)
    prec = rp.get("precision")
    recall = rp.get("recall")
    f1 = rp.get("f1")
    dislike_src_rate = rp.get("dislike_rate_sourced")

    col_pos = "overlap match" if overlap else "has source"
    col_neg = "no match" if overlap else "no source"
    cm_table = (
        "<table style='width:auto;margin:8px 0'>"
        "<tr><th></th>"
        f"<th style='text-align:center;padding:4px 12px'>{col_pos}<br>"
        "<span class='muted' style='font-weight:normal;font-size:.8em'>ŷ = 1</span></th>"
        f"<th style='text-align:center;padding:4px 12px'>{col_neg}<br>"
        "<span class='muted' style='font-weight:normal;font-size:.8em'>ŷ = 0</span></th>"
        "</tr>"
        f"<tr><td><b>liked</b> <span class='muted' style='font-size:.8em'>y=1</span></td>"
        f"<td style='text-align:center;background:#d4edda;color:#155724;font-weight:600;padding:6px 12px'>TP&nbsp;{tp:,}</td>"
        f"<td style='text-align:center;background:#e8f4ea;color:#4a7c59;padding:6px 12px'>TN&nbsp;{tn:,}</td></tr>"
        f"<tr><td><b>disliked</b> <span class='muted' style='font-size:.8em'>y=0</span></td>"
        f"<td style='text-align:center;background:#f8d7da;color:#721c24;font-weight:600;padding:6px 12px'>FP&nbsp;{fp:,}</td>"
        f"<td style='text-align:center;background:#fff3e0;color:#a04000;padding:6px 12px'>FN&nbsp;{fn:,}</td></tr>"
        f"<tr><td><span class='muted'>unrated</span></td>"
        f"<td style='text-align:center;color:#aaa;font-size:.85em;padding:4px 12px'>{unrated_src:,}</td>"
        f"<td style='text-align:center;color:#aaa;font-size:.85em;padding:4px 12px'>{unrated_no:,}</td>"
        "</tr></table>"
    )

    pr_summary = ""
    if prec is not None and recall is not None:
        f1_str = f"&nbsp;·&nbsp; <b>F1 {f1:.1%}</b>" if f1 is not None else ""
        pr_summary = (
            f"<p style='margin:4px 0'>"
            f"Precision <b>{prec:.1%}</b>"
            f" <span class='muted' style='font-size:.85em'>= TP/(TP+FP)</span>"
            f" &nbsp;·&nbsp; Recall <b>{recall:.1%}</b>"
            f" <span class='muted' style='font-size:.85em'>= TP/(TP+TN−esc) = liked_sourced/n_liked</span>"
            f"{f1_str}</p>"
        )

    dlr = s.get("dislike_like_ratio")
    bias_note = ""
    if dlr is not None and dlr > 2.0:
        bias_note = (
            "<p class='muted' style='font-size:.78em;margin:2px 0 6px'>"
            f"⚠ Eval dislike:like = {dlr:.1f}:1 (natural ~2:1) — "
            "FP count is inflated by sampling; precision is biased low vs production."
            "</p>"
        )

    inversion_note = ""
    if rp.get("sources_outpace_satisfaction"):
        inversion_note = (
            "<div style='background:#fff3cd;border:1px solid #ffc107;"
            "border-radius:4px;padding:8px 12px;margin-top:8px;font-size:.85em'>"
            "<b>⚠ Hit rate inversion:</b> disliked responses cited sources at a "
            "<i>higher</i> rate than liked ones — "
            "this is the core case for grounding evals."
            "</div>"
        )
    if prec is not None and dislike_src_rate is not None and dislike_src_rate > prec:
        inversion_note += (
            "<div style='background:#fff3cd;border:1px solid #ffc107;"
            "border-radius:4px;padding:8px 12px;margin-top:8px;font-size:.85em'>"
            f"<b>⚠ FP &gt; TP among rated sourced turns</b> "
            f"(precision {prec:.0%}) — citation quality likely the issue; run grounding evals."
            "</div>"
        )

    return (
        f"<h3>{_html.escape(title)}</h3>"
        f"<p class='muted' style='margin:0 0 4px;font-size:.85em'>{y_note}</p>"
        f"<p class='muted' style='margin:0 0 4px'>Unique URLs cited: <b>{rp.get('unique_urls_cited', 0)}</b></p>"
        "<table><tr><th>Sentiment</th><th>n</th><th>Source hit rate</th><th></th></tr>"
        f"{hit_rows}</table>"
        f"{cm_table}"
        f"{pr_summary}"
        f"{bias_note}"
        f"{inversion_note}"
        "<p class='muted' style='font-size:.78em;margin:6px 0 0'>"
        "Layer-2 grounding grader pass rates are in the judge table above/below — "
        "not this citation proxy.</p>"
        "<!-- end-retrieval-proxy -->"
    )


def _content_profile_block(stats: dict) -> str:
    qs = stats.get("qa_stats", {})
    ts = stats.get("turn_stats", {})
    src_dist = stats.get("retrieval_proxy", {}).get("source_count_distribution", {})
    rp = stats.get("retrieval_proxy", {})

    def _src_cells(v_src, v_no_src, fmt="{:.1f}") -> str:
        sv = fmt.format(v_src) if v_src is not None else "—"
        nv = fmt.format(v_no_src) if v_no_src is not None else "—"
        return f"<td class='num muted'>{sv}</td><td class='num muted'>{nv}</td>"

    stat_rows = (
        f"<tr><td>Query length (words)</td>"
        f"<td class='num'>{qs.get('min_query_words', 0)}</td>"
        f"<td class='num'><b>{qs.get('avg_query_words', 0):.1f}</b></td>"
        + _src_cells(qs.get('avg_query_words_sourced'), qs.get('avg_query_words_unsourced'))
        + f"<td class='num'>{qs.get('max_query_words', 0)}</td></tr>"
        f"<tr><td>Response length (words)</td>"
        f"<td class='num'>{qs.get('min_response_words', 0)}</td>"
        f"<td class='num'><b>{qs.get('avg_response_words', 0):.1f}</b></td>"
        + _src_cells(qs.get('avg_response_words_sourced'), qs.get('avg_response_words_unsourced'))
        + f"<td class='num'>{qs.get('max_response_words', 0)}</td></tr>"
    )
    if ts.get("avg_turn_count") is not None:
        stat_rows += (
            f"<tr><td>Conv turns</td>"
            f"<td class='num'>{ts.get('min_turn_count', 0)}</td>"
            f"<td class='num'><b>{ts['avg_turn_count']:.1f}</b></td>"
            + _src_cells(ts.get('avg_turn_count_sourced_conv'), ts.get('avg_turn_count_unsourced_conv'))
            + f"<td class='num'>{ts.get('max_turn_count', 0)}</td></tr>"
        )
    if ts.get("avg_duration_sec") is not None:
        stat_rows += (
            f"<tr><td>Conv duration (min)</td>"
            f"<td class='num'>{ts.get('min_duration_sec', 0) / 60:.1f}</td>"
            f"<td class='num'><b>{ts['avg_duration_sec'] / 60:.1f}</b></td>"
            + _src_cells(
                ts['avg_duration_sec_sourced_conv'] / 60 if ts.get('avg_duration_sec_sourced_conv') is not None else None,
                ts['avg_duration_sec_unsourced_conv'] / 60 if ts.get('avg_duration_sec_unsourced_conv') is not None else None,
            )
            + f"<td class='num'>{ts.get('max_duration_sec', 0) / 60:.1f}</td></tr>"
        )
    if ts.get("avg_turn_interval_sec") is not None:
        stat_rows += (
            f"<tr><td>Turn interval (s)</td>"
            f"<td class='num'>{ts.get('min_turn_interval_sec', 0):.0f}</td>"
            f"<td class='num'><b>{ts['avg_turn_interval_sec']:.0f}</b></td>"
            + _src_cells(ts.get('avg_turn_interval_sec_sourced_conv'), ts.get('avg_turn_interval_sec_unsourced_conv'), fmt="{:.0f}")
            + f"<td class='num'>{ts.get('max_turn_interval_sec', 0):.0f}</td></tr>"
        )
    if qs.get("max_source_count") is not None:
        _asg = qs.get("avg_source_count_given_sourced")
        stat_rows += (
            f"<tr><td>Sources (per response)</td>"
            f"<td class='num'>{qs.get('min_source_count', 0)}</td>"
            f"<td class='num'><b>{qs.get('avg_source_count', 0):.2f}</b></td>"
            f"<td class='num muted'>{f'{_asg:.2f}' if _asg is not None else '—'}</td>"
            f"<td class='num muted'>—</td>"
            f"<td class='num'>{qs.get('max_source_count', 0)}</td></tr>"
        )

    mhs = qs.get("mean_has_source_liked")
    mhs_note = f" &nbsp;·&nbsp; sourced when liked: <b>{mhs:.0%}</b>" if mhs is not None else ""
    avg_src_liked = rp.get("avg_sources_liked")
    avg_src_liked_note = f" &nbsp;·&nbsp; avg src/liked: <b>{avg_src_liked:.2f}</b>" if avg_src_liked is not None else ""
    footnotes = ""
    if qs.get("avg_response_chars"):
        footnotes += (
            f"<p class='muted' style='margin:4px 0 0'>Avg response: <b>{qs['avg_response_chars']:,}</b> chars"
            f" &nbsp;·&nbsp; avg sources/turn: <b>{qs.get('avg_source_count', 0):.2f}</b>"
            f"{mhs_note}{avg_src_liked_note}</p>"
        )
    if src_dist:
        src_dist_str = " &nbsp; ".join(f"<b>{k}</b> src: {v}" for k, v in sorted(src_dist.items()))
        footnotes += f"<p class='muted' style='margin:2px 0 0;font-size:.85em'>{src_dist_str}</p>"
    if ts.get("avg_turns_to_first_like") is not None:
        footnotes += f"<p class='muted' style='margin:2px 0 0'>Turns to first like: <b>{ts['avg_turns_to_first_like']:.1f}</b> avg</p>"
    dist = ts.get("turn_count_dist", {})
    if dist:
        bucket_order = ["1", "2–3", "4–6", "7+"]
        dist_str = " &nbsp; ".join(f"<b>{k}</b>: {dist[k]}" for k in bucket_order if k in dist)
        footnotes += f"<p class='muted' style='margin:2px 0 0;font-size:.85em'>Turn dist: {dist_str}</p>"

    return (
        "<h3>Content &amp; turn profile</h3>"
        "<table><tr><th>Metric</th><th class='num'>Min</th><th class='num'>Mean</th>"
        "<th class='num' title='mean: turns with sources'><span class='muted'>+src</span></th>"
        "<th class='num' title='mean: turns without sources'><span class='muted'>-src</span></th>"
        "<th class='num'>Max</th></tr>"
        f"{stat_rows}</table>"
        f"{footnotes}"
    )


def _population_block(stats: dict) -> str:
    lang_bk = stats.get("language_breakdown", {})
    co_bk = stats.get("conv_outcome_breakdown", {})
    if not lang_bk and not co_bk:
        return ""

    s = stats["sentiment"]
    cov = s.get("rating_coverage_rate", 0.0)
    uncl = stats.get("unclassified_disliked", {})
    eg_share = stats.get("e_grounding_share_of_sourced_disliked", 0.0)

    lang_col = ""
    if lang_bk:
        lang_total = sum(lang_bk.values())
        ml_stats = stats.get("multilingual_stats", {})
        ml_by_lang = ml_stats.get("by_language", {})
        has_ml_col = any(ml_by_lang.get(lg, 0) > 0 for lg in lang_bk)
        lang_rows = ""
        for lg, cnt in lang_bk.items():
            ml_cell = f"<td class='num'>{ml_by_lang.get(lg, 0):.0%}</td>" if has_ml_col else ""
            lang_rows += (
                f"<tr><td>{lg}</td><td class='num'>{cnt:,}</td>"
                f"<td class='num'>{cnt / lang_total:.0%}</td>{ml_cell}</tr>"
            )
        ml_header = "<th>Multilingual</th>" if has_ml_col else ""
        cov_note = (
            f"<p class='muted' style='margin:4px 0 0'>Rating coverage: <b>{cov:.1%}</b> of turns rated</p>"
        ) if cov > 0 else ""
        ml_legend = ""
        if has_ml_col:
            ml_legend = (
                f"<p class='muted' style='font-size:.78em;margin:4px 0 0'>"
                f"Multilingual = % of convs per language that switched query language mid-conv"
                f" &nbsp;·&nbsp; All convs: <b>{ml_stats.get('rate', 0):.0%}</b>"
                f" &nbsp;·&nbsp; Turns with lang-change: <b>{ml_stats.get('change_turn_rate', 0):.0%}</b>"
                f"</p>"
            )
        lang_detect_note = (
            "<p class='muted' style='font-size:.78em;margin:6px 0 0'>"
            "<b>Caveat:</b> Lang codes come from <code>langdetect</code> on query text — "
            "Danish is often confused with other Scandinavian languages or labeled "
            "<code>unknown</code> on short queries. Directional only.</p>"
        )
        lang_col = (
            "<div><h3>Language breakdown</h3>"
            f"<table><tr><th>Lang</th><th>n</th><th>%</th>{ml_header}</tr>"
            f"{lang_rows}</table>{lang_detect_note}{cov_note}{ml_legend}</div>"
        )

    co_col = ""
    if co_bk:
        co_total = sum(co_bk.values())
        co_sent_bk = stats.get("conv_outcome_sentiment_breakdown", {})
        co_rows = ""
        for outcome, cnt in co_bk.items():
            sent = co_sent_bk.get(outcome, {})
            n_liked = sent.get("liked", 0)
            n_disliked = sent.get("disliked", 0)
            n_unrated = sent.get("unrated", 0)

            def _sc(n: int) -> str:
                return (
                    f"<td class='num muted'>{n:,}</td>" if n
                    else "<td class='num muted' style='color:#ccc'>—</td>"
                )
            co_rows += (
                f"<tr><td>{outcome}</td><td class='num'>{cnt:,}</td>"
                f"<td class='num'>{cnt / co_total:.0%}</td>"
                + _sc(n_liked) + _sc(n_disliked) + _sc(n_unrated)
                + f"<td>{_pct_bar(cnt / co_total, 60, '#8ab4f8')}</td></tr>"
            )
        notes = ""
        if uncl.get("count", 0) > 0:
            notes += (
                f"<p class='muted' style='margin:4px 0 0'>Unclassified disliked: "
                f"<b>{uncl['count']}</b> ({uncl['rate']:.0%} of disliked — taxonomy gap)</p>"
            )
        if eg_share > 0:
            notes += (
                f"<p class='muted' style='margin:4px 0 0'>"
                f"<b>E_grounding</b> = {eg_share:.0%} of sourced-disliked turns "
                f"<span style='font-size:.78em'>"
                f"(by definition: E_grounding ≡ disliked + has_source — "
                f"confirms label consistency, not a finding)</span></p>"
            )
        co_legend = (
            "<p class='muted' style='font-size:.78em;margin:4px 0 0'>"
            "<b>resolved:</b> ≥1 like, no friction signals &nbsp;·&nbsp; "
            "<b>resolved_with_friction:</b> liked but also has dislike, escalation, or repeated query &nbsp;·&nbsp; "
            "<b>unresolved:</b> dislikes only, no likes &nbsp;·&nbsp; "
            "<b>unrated:</b> no ratings in conv</p>"
        )
        co_col = (
            "<div>"
            "<h3>Conversation outcomes <span class='muted' style='font-size:.8em'>(unique convs)</span></h3>"
            "<table><tr><th>Outcome</th><th>n convs</th><th>%</th>"
            "<th class='num muted' style='font-size:.85em'>liked↑</th>"
            "<th class='num muted' style='font-size:.85em'>disliked↓</th>"
            "<th class='num muted' style='font-size:.85em'>unrated</th>"
            "<th></th></tr>"
            f"{co_rows}</table>{notes}{co_legend}</div>"
        )

    return (
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;"
        "margin-top:16px;padding-top:12px;border-top:1px solid #e8e8e8'>"
        f"{lang_col}{co_col}</div>"
    )


def _qa_pairs_block(qa_samples: dict[str, list[dict]], ft_order: list[str]) -> str:
    if not qa_samples:
        return ""
    cards = ""
    for s in ("liked", "disliked", "unrated"):
        pairs = qa_samples.get(s)
        if not pairs:
            continue
        cards += f"<h3>{s}</h3>"
        for pair in pairs:
            ft = pair.get("ft", "")
            ft_badge = (
                f' <span class="chip" style="background:#f0f0f0;color:#666;font-size:.7em">{ft}</span>'
                if ft and ft not in ("no_failure", "n/a") else ""
            )
            q = _html.escape(pair["query"][:220])
            r = _html.escape(pair["response"][:350])
            url_links = ""
            for url in pair.get("urls", [])[:3]:
                eu = _html.escape(url)
                url_links += (
                    f'<a href="{eu}" target="_blank" '
                    f'style="color:#2563eb;font-size:.8em;margin-right:8px">{eu[:80]}</a>'
                )
            sources_block = (
                f'<div style="margin-top:4px">{url_links}</div>' if url_links else ""
            )
            cards += (
                f'<div class="qa-card">'
                f'<div class="qa-label"><span class="chip {pair["sentiment"]}">{pair["sentiment"]}</span>{ft_badge}</div>'
                f'<div class="qa-q">Q: {q}</div>'
                f'<div class="qa-r">A: {r}{"…" if len(pair["response"]) > 350 else ""}</div>'
                f"{sources_block}"
                f"</div>"
            )
    return (
        '<div style="margin-top:20px;border-top:1px solid #e0e0e0;padding-top:16px">'
        "<h3>Sample QA pairs</h3>"
        f"{cards}</div>"
    ) if cards else ""


def _url_coverage_section(cov: dict) -> str:
    if not cov:
        return ""
    total = cov.get("total_citations", 0)
    unique = cov.get("unique_urls", 0)
    once = cov.get("cited_once", 0)
    five_plus = cov.get("cited_5plus", 0)

    top_rows = "".join(
        f"<tr><td><a href='{_html.escape(r['url'])}' target='_blank' style='color:#2563eb'>"
        f"{_html.escape(r['url'][:80])}</a></td>"
        f"<td class='num'>{r['count']}</td>"
        f"<td>{_pct_bar(r['count'] / max(cov['top_cited'][0]['count'], 1), 80, '#4c9be8')}</td></tr>"
        for r in cov.get("top_cited", [])
    )
    bottom_rows = "".join(
        f"<tr><td><a href='{_html.escape(r['url'])}' target='_blank' style='color:#888'>"
        f"{_html.escape(r['url'][:80])}</a></td>"
        f"<td class='num'>{r['count']}</td></tr>"
        for r in cov.get("bottom_cited", [])
    )

    art_cov = cov.get("article_coverage", {})
    art_total = art_cov.get("total_articles", 0)
    art_with_url = art_cov.get("articles_with_url", 0)
    art_covered = art_cov.get("covered", 0)
    art_uncovered = art_cov.get("uncovered", 0)
    art_status = art_cov.get("status", "")
    domain_mismatch = art_cov.get("domain_mismatch", False)
    kb_domains = art_cov.get("kb_domains", [])
    cited_domains = art_cov.get("cited_domains", [])

    domain_warning = ""
    if domain_mismatch:
        kb_str = ", ".join(kb_domains[:3]) or "unknown"
        cited_str = ", ".join(cited_domains[:3]) or "unknown"
        domain_warning = (
            "<div style='background:#fff3cd;border:1px solid #ffc107;"
            "border-radius:4px;padding:8px 12px;margin:8px 0;font-size:.85em'>"
            f"<b>⚠ URL domain mismatch:</b> KB articles are on <code>{_html.escape(kb_str)}</code> "
            f"— agent citations are on <code>{_html.escape(cited_str)}</code>. "
            "These are the same articles on different domains (e.g. help.shine.co ↔ billy.dk/support). "
            "Coverage matching uses article IDs extracted from URLs — 0% coverage is an artifact "
            "if the ID pattern differs between domains, not a real KB gap."
            "</div>"
        )

    kb_line = ""
    if art_total:
        if art_status == "ok" and art_with_url:
            kb_line = (
                f"<p class='muted' style='margin:4px 0 0'>"
                f"KB articles: <b>{art_total:,}</b> stored &nbsp;·&nbsp; "
                f"<b>{art_with_url:,}</b> with URL &nbsp;·&nbsp; "
                f"<b style='color:#155724'>{art_covered}</b> cited &nbsp;·&nbsp; "
                f"<b style='color:#721c24'>{art_uncovered}</b> never cited &nbsp;·&nbsp; "
                f"coverage: <b>{art_covered / art_with_url:.0%}</b>"
                f"</p>"
            )
        else:
            kb_line = (
                f"<p class='muted' style='margin:4px 0 0'>"
                f"KB articles: <b>{art_total:,}</b> stored"
                f"{'&nbsp;·&nbsp; URL metadata not yet available' if art_status == 'metadata_tbd' else ''}"
                f"</p>"
            )

    uncovered_items = art_cov.get("uncovered_items", [])
    uncited_rows = "".join(
        f"<tr><td><a href='{_html.escape(r['url'])}' target='_blank' style='color:#888'>"
        f"{_html.escape(r['url'][:100])}</a></td>"
        f"<td class='num muted' style='color:#bbb'>0</td></tr>"
        for r in uncovered_items
    )
    uncited_block = ""
    if uncited_rows:
        uncited_block = (
            f"<details class='methodology-fold' style='margin-top:16px'>"
            f"<summary>Uncited KB articles ({art_uncovered}) — long tail of Intercom corpus</summary>"
            f"<table class='detail-table' style='margin-top:8px'>"
            f"<tr><th>URL</th><th class='num'>Citations</th></tr>{uncited_rows}</table>"
            f"</details>"
        )

    return (
        f'<div class="section">'
        f"<h2>URL Citation Coverage</h2>"
        f"<p><b>{total:,}</b> total citations &nbsp;·&nbsp; "
        f"<b>{unique:,}</b> distinct URLs cited &nbsp;·&nbsp; "
        f"long-tail (cited only ×1): <b>{once}</b> &nbsp;·&nbsp; cited ×5+: <b>{five_plus}</b></p>"
        f"<p class='muted' style='font-size:.78em;margin:2px 0 4px'>"
        f"<i>Distinct URLs</i> = unique article/page URLs across all responses. "
        f"<i>Long-tail</i> = subset cited in exactly 1 response.</p>"
        f"{kb_line}"
        f"{domain_warning}"
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px'>"
        f"<div><h3>Top 20 cited URLs</h3>"
        f"<table><tr><th>URL</th><th>Citations</th><th></th></tr>{top_rows}</table></div>"
        f"<div><h3>Bottom 20 cited URLs (long tail)</h3>"
        f"<table><tr><th>URL</th><th>Citations</th></tr>{bottom_rows}</table></div>"
        f"</div>"
        f"{uncited_block}"
        f"</div>"
    )


def _section(
    label: str,
    stats: dict,
    qa_samples: dict | None,
    ft_order: list[str],
) -> str:
    """Assemble a full per-file section block from sub-builders."""
    cats = list(stats["categories"].items())[:10]

    left = (
        _failure_type_block(stats["failure_types"])
        + _response_type_block(stats["response_types"])
        + _content_profile_block(stats)
    )
    right = _category_block(cats, stats) + _retrieval_proxy_block(stats)

    qa_block = _qa_pairs_block(qa_samples, ft_order) if qa_samples else ""

    return (
        f'<div class="section">'
        f"<h2>{label}</h2>{_summary_header(stats)}"
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px'>"
        f"<div>{left}</div>"
        f"<div>{right}</div>"
        f"</div>"
        f"{_population_block(stats)}"
        f"{qa_block}"
        f"</div>"
    )
