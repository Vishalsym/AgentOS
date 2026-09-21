"""An intentionally empty scenario: a fresh kernel with no scripted agents,
for demos where the dashboard should reflect only real, externally driven
activity -- e.g. the MCP bridge relaying calls from a live Claude chat."""

from __future__ import annotations

import os

from sentari.kernel import Kernel

from .state import DashboardState, build_default_notifier, build_default_provider


async def run_live_scenario(state: DashboardState) -> None:
    provider = build_default_provider()
    using_real_api = bool(os.environ.get("ANTHROPIC_API_KEY"))
    # Token-aware quota only when the real API is actually in use: with the
    # offline mock, quota stays the familiar flat call-count model so the
    # dashboard's default (no key configured) experience is unchanged --
    # nobody sees an unexpected "truncated" warning from a quota=5 default
    # meant as "5 calls," not "5 tokens".
    state.kernel = Kernel(
        notifier=build_default_notifier(), provider=provider, token_aware_quota=using_real_api
    )
    await state.narrate(
        "Live mode -- no scripted agents. Every agent, resource, and log line below is real, "
        "driven by whatever calls the kernel right now (e.g. the MCP bridge or the Cockpit)."
        + (
            " ANTHROPIC_API_KEY is set: tool_call prompts hit the real Claude API, and each "
            "agent's quota is now a real TOKEN budget (input+output), not a call count -- a "
            "prompt that would exceed it is cut off at the cap and returned as a flagged "
            "partial result, not silently denied."
            if using_real_api
            else " (Set ANTHROPIC_API_KEY before starting the dashboard to use the real Claude "
            "API and real token-based quota instead of the offline mock/call-count model.)"
        ),
        "live",
    )
    await state.finish()
