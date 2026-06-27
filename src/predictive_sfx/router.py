from __future__ import annotations

REALTIME_TOLERANT = {
    "ambient", "ambience", "music", "wind", "rain", "drone", "hum", "water",
    "ui", "menu", "voice", "dialogue", "weather",
}


class EventRouter:
    def __init__(self, realtime_events=None, default: str = "prefetch"):
        self.realtime_events = (set(realtime_events) if realtime_events is not None
                                else set(REALTIME_TOLERANT))
        self.default = default

    def route(self, event_type: str) -> str:
        return "realtime" if event_type in self.realtime_events else self.default

    def is_prefetch(self, event_type: str) -> bool:
        return self.route(event_type) == "prefetch"

    def is_realtime(self, event_type: str) -> bool:
        return self.route(event_type) == "realtime"
