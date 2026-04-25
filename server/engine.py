from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


RESOURCE_TYPES = ("trucks", "medical_teams", "food", "rescue_units")


@dataclass
class ZoneState:
    zone_id: str
    severity: float
    requirements: Dict[str, int]
    delivered: Dict[str, int]
    road_open: bool
    active: bool = True

    def progress_ratio(self) -> float:
        required_total = sum(max(v, 0) for v in self.requirements.values())
        if required_total == 0:
            return 1.0
        delivered_total = 0
        for key in self.requirements:
            delivered_total += min(self.delivered.get(key, 0), self.requirements.get(key, 0))
        return max(0.0, min(1.0, delivered_total / required_total))

    def is_fulfilled(self) -> bool:
        for resource, needed in self.requirements.items():
            if self.delivered.get(resource, 0) < needed:
                return False
        return True


@dataclass
class InTransitOrder:
    zone_id: str
    resource_type: str
    amount: int
    eta: int


@dataclass
class SimulationState:
    task_id: str
    tick: int
    max_steps: int
    zones: Dict[str, ZoneState]
    depot: Dict[str, int]
    in_transit: List[InTransitOrder] = field(default_factory=list)
    task_score: float = 0.5
    logs: List[str] = field(default_factory=list)


TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "task_easy_local": {
        "seed": 11,
        "max_steps": 20,
        "zones": 4,
        "depot": {"trucks": 10, "medical_teams": 8, "food": 24, "rescue_units": 6},
        "scripted_events": {},
        "dynamic_events": False,
    },
    "task_medium_city": {
        "seed": 23,
        "max_steps": 28,
        "zones": 7,
        "depot": {"trucks": 14, "medical_teams": 10, "food": 34, "rescue_units": 8},
        "scripted_events": {
            4: [{"type": "block_road", "zone_id": "zone_3"}],
            7: [{"type": "priority_spike", "zone_id": "zone_5", "delta": 0.18}],
        },
        "dynamic_events": False,
    },
    "task_hard_region": {
        "seed": 47,
        "max_steps": 36,
        "zones": 12,
        "depot": {"trucks": 18, "medical_teams": 14, "food": 42, "rescue_units": 10},
        "scripted_events": {
            3: [{"type": "depot_depletion", "resource_type": "food", "amount": 4}],
            8: [{"type": "priority_spike", "zone_id": "zone_9", "delta": 0.22}],
            12: [{"type": "block_road", "zone_id": "zone_2"}],
        },
        "dynamic_events": True,
    },
}


class DisasterReliefEngine:
    def __init__(self) -> None:
        self._rng = random.Random()
        self._state: SimulationState | None = None

    def reset(self, task_id: str) -> Dict[str, Any]:
        config = TASK_CONFIGS.get(task_id, TASK_CONFIGS["task_easy_local"])
        self._rng.seed(config["seed"])
        zones = self._build_zones(config["zones"])
        self._state = SimulationState(
            task_id=task_id if task_id in TASK_CONFIGS else "task_easy_local",
            tick=0,
            max_steps=config["max_steps"],
            zones=zones,
            depot=copy.deepcopy(config["depot"]),
        )
        self._state.logs.append(f"Simulation reset for {self._state.task_id}.")
        return self._observation(include_score=False)

    def step(self, command: str) -> Tuple[Dict[str, Any], float, bool]:
        if self._state is None:
            self.reset("task_easy_local")

        assert self._state is not None
        command_feedback, action_penalty = self._apply_command(command)
        self._state.logs.append(command_feedback)

        tick_reward = self._advance_tick()
        reward = self._clamp_reward(0.5 + tick_reward - action_penalty)
        self._state.task_score = self._clamp_score(self._state.task_score + (reward - 0.5) * 0.35)

        done = self._is_done()
        if done:
            self._apply_terminal_penalties()
            reward = self._clamp_reward(self._state.task_score)

        observation = self._observation(include_score=True)
        return observation, reward, done

    def _build_zones(self, count: int) -> Dict[str, ZoneState]:
        zones: Dict[str, ZoneState] = {}
        for i in range(1, count + 1):
            zone_id = f"zone_{i}"
            severity = round(self._rng.uniform(0.28, 0.82), 3)
            requirements = {
                "trucks": self._rng.randint(1, 3),
                "medical_teams": self._rng.randint(1, 3),
                "food": self._rng.randint(2, 7),
                "rescue_units": self._rng.randint(1, 2),
            }
            zones[zone_id] = ZoneState(
                zone_id=zone_id,
                severity=severity,
                requirements=requirements,
                delivered={k: 0 for k in RESOURCE_TYPES},
                road_open=True,
            )
        return zones

    def _apply_command(self, command: str) -> Tuple[str, float]:
        cmd = command.strip().lower()
        if cmd == "get_status":
            return self._status_text(), 0.0

        if cmd == "check_routes":
            return self._route_text(), 0.0

        if cmd == "wait":
            return "Advance one tick without new deployments.", 0.01

        match = re.match(r"deploy\s+([a-z_]+)\s+(\d+)\s+to\s+([a-z0-9_]+)$", cmd)
        if match:
            resource_type = match.group(1)
            amount = int(match.group(2))
            zone_id = match.group(3)
            return self._handle_deploy(resource_type, amount, zone_id)

        return (
            "Invalid command. Use get_status, deploy <resource_type> <amount> to <zone_id>, check_routes, or wait.",
            0.08,
        )

    def _handle_deploy(self, resource_type: str, amount: int, zone_id: str) -> Tuple[str, float]:
        assert self._state is not None
        if resource_type not in RESOURCE_TYPES:
            return f"Unknown resource type '{resource_type}'.", 0.06
        if amount <= 0:
            return "Deployment amount must be positive.", 0.06
        zone = self._state.zones.get(zone_id)
        if zone is None:
            return f"Unknown zone '{zone_id}'.", 0.06
        if not zone.road_open:
            return f"Road to {zone_id} is blocked. Deployment failed.", 0.07
        if self._state.depot[resource_type] < amount:
            return f"Insufficient {resource_type} at depot.", 0.05

        self._state.depot[resource_type] -= amount
        self._state.in_transit.append(
            InTransitOrder(zone_id=zone_id, resource_type=resource_type, amount=amount, eta=self._state.tick + 1)
        )
        return f"Deployment queued: {amount} {resource_type} to {zone_id}.", 0.0

    def _advance_tick(self) -> float:
        assert self._state is not None
        self._state.tick += 1
        event_bonus = 0.0

        # Apply transit arrivals.
        remaining_orders: List[InTransitOrder] = []
        for order in self._state.in_transit:
            if order.eta <= self._state.tick:
                zone = self._state.zones[order.zone_id]
                if zone.road_open:
                    zone.delivered[order.resource_type] += order.amount
                    event_bonus += 0.02
                else:
                    self._state.depot[order.resource_type] += order.amount
                    event_bonus -= 0.02
                    self._state.logs.append(f"Order to {order.zone_id} bounced back due to blocked road.")
            else:
                remaining_orders.append(order)
        self._state.in_transit = remaining_orders

        # Degrade and stabilize zones.
        progress_bonus = 0.0
        inefficiency_penalty = 0.0
        for zone in self._state.zones.values():
            prior_progress = zone.progress_ratio()
            if zone.is_fulfilled():
                zone.severity = max(0.01, zone.severity - 0.15)
                progress_bonus += 0.12
            else:
                zone.severity = min(0.99, zone.severity + 0.04)

            current_progress = zone.progress_ratio()
            progress_bonus += max(0.0, current_progress - prior_progress) * 0.15

            # Over-supplying is treated as inefficiency.
            for resource in RESOURCE_TYPES:
                surplus = zone.delivered[resource] - zone.requirements[resource]
                if surplus > 0:
                    inefficiency_penalty += 0.005 * surplus

            zone.active = zone.severity > 0.06

        self._apply_events()
        return progress_bonus + event_bonus - inefficiency_penalty

    def _apply_events(self) -> None:
        assert self._state is not None
        config = TASK_CONFIGS[self._state.task_id]

        scripted = config.get("scripted_events", {}).get(self._state.tick, [])
        for event in scripted:
            self._trigger_event(event)

        if config.get("dynamic_events"):
            roll = self._rng.random()
            if roll < 0.24:
                event = self._random_event()
                self._trigger_event(event)

    def _random_event(self) -> Dict[str, Any]:
        assert self._state is not None
        zone_id = self._rng.choice(list(self._state.zones.keys()))
        event_type = self._rng.choice(["block_road", "priority_spike", "depot_depletion", "reopen_road"])
        if event_type == "depot_depletion":
            resource = self._rng.choice(list(RESOURCE_TYPES))
            return {"type": "depot_depletion", "resource_type": resource, "amount": self._rng.randint(1, 3)}
        if event_type == "priority_spike":
            return {"type": "priority_spike", "zone_id": zone_id, "delta": round(self._rng.uniform(0.08, 0.2), 3)}
        return {"type": event_type, "zone_id": zone_id}

    def _trigger_event(self, event: Dict[str, Any]) -> None:
        assert self._state is not None
        event_type = event.get("type")
        if event_type == "block_road":
            zone = self._state.zones.get(event.get("zone_id"))
            if zone is not None:
                zone.road_open = False
                self._state.logs.append(f"Event: road blocked to {zone.zone_id}.")
            return
        if event_type == "reopen_road":
            zone = self._state.zones.get(event.get("zone_id"))
            if zone is not None:
                zone.road_open = True
                self._state.logs.append(f"Event: road reopened to {zone.zone_id}.")
            return
        if event_type == "depot_depletion":
            resource = event.get("resource_type")
            amount = int(event.get("amount", 1))
            if resource in RESOURCE_TYPES:
                self._state.depot[resource] = max(0, self._state.depot[resource] - amount)
                self._state.logs.append(f"Event: depot {resource} depleted by {amount}.")
            return
        if event_type == "priority_spike":
            zone = self._state.zones.get(event.get("zone_id"))
            if zone is not None:
                delta = float(event.get("delta", 0.1))
                zone.severity = min(0.99, zone.severity + delta)
                self._state.logs.append(f"Event: priority spike in {zone.zone_id} (+{delta:.2f} severity).")

    def _is_done(self) -> bool:
        assert self._state is not None
        if self._state.tick >= self._state.max_steps:
            return True
        return all((zone.is_fulfilled() and zone.severity <= 0.08) for zone in self._state.zones.values())

    def _apply_terminal_penalties(self) -> None:
        assert self._state is not None
        abandoned = 0
        for zone in self._state.zones.values():
            delivered_total = sum(zone.delivered.values())
            if zone.active and delivered_total == 0:
                abandoned += 1
        if abandoned > 0:
            self._state.task_score = self._clamp_score(self._state.task_score - 0.45 - 0.08 * abandoned)
            self._state.logs.append(f"Terminal penalty: {abandoned} active zones abandoned.")

    def _status_text(self) -> str:
        assert self._state is not None
        top_zones = sorted(self._state.zones.values(), key=lambda z: z.severity, reverse=True)[:4]
        zone_summary = ", ".join(f"{z.zone_id}:{z.severity:.2f}" for z in top_zones)
        depot_summary = ", ".join(f"{k}={v}" for k, v in self._state.depot.items())
        return f"Tick {self._state.tick} | Hot zones [{zone_summary}] | Depot [{depot_summary}]"

    def _route_text(self) -> str:
        assert self._state is not None
        route_summary = ", ".join(
            f"{z.zone_id}={'open' if z.road_open else 'blocked'}" for z in self._state.zones.values()
        )
        return f"Route status: {route_summary}"

    def _observation(self, include_score: bool) -> Dict[str, Any]:
        assert self._state is not None
        zones_payload = {
            zone.zone_id: {
                "severity": round(zone.severity, 3),
                "requirements": zone.requirements,
                "delivered": zone.delivered,
                "road_open": zone.road_open,
                "active": zone.active,
                "progress": round(zone.progress_ratio(), 3),
            }
            for zone in self._state.zones.values()
        }
        observation = {
            "task_id": self._state.task_id,
            "tick": self._state.tick,
            "max_steps": self._state.max_steps,
            "zones": zones_payload,
            "depot": self._state.depot,
            "in_transit": [order.__dict__ for order in self._state.in_transit],
            "recent_logs": self._state.logs[-8:],
        }
        if include_score:
            observation["task_score"] = round(self._state.task_score, 6)
        return observation

    @staticmethod
    def _clamp_score(value: float) -> float:
        return max(0.001, min(0.999, value))

    @staticmethod
    def _clamp_reward(value: float) -> float:
        return max(0.001, min(0.999, value))
