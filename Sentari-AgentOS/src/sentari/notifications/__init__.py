"""Notification sinks for significant kernel-level decisions (agent killed,
deadlock victim selected, quota exhausted) -- the README roadmap's
"dashboard/email notifications" item. A notification is best-effort and
observational: a sink failure must never affect the kernel decision that
triggered it, exactly like a single agent's failure must never crash the
kernel (SyscallLayer.on_syscall's containment philosophy, applied here to
the notification path instead of a tool call)."""

from .notifier import CompositeNotifier, LogNotifier, NotificationEvent, Notifier, WebhookNotifier

__all__ = [
    "CompositeNotifier",
    "LogNotifier",
    "NotificationEvent",
    "Notifier",
    "WebhookNotifier",
]
