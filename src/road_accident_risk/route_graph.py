"""Directed OSM routing and additive costs for the route-proxy experiment."""

from __future__ import annotations

from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

BASE_SPEEDS_KMH = {
    "motorway": 90.0, "motorway_link": 45.0,
    "trunk": 70.0, "trunk_link": 40.0,
    "primary": 55.0, "primary_link": 30.0,
    "secondary": 45.0, "secondary_link": 25.0,
    "tertiary": 35.0, "tertiary_link": 20.0,
    "unclassified": 25.0, "residential": 25.0, "living_street": 10.0,
    "service": 15.0,
}
SERVICE_SPEEDS_KMH = {
    "driveway": 10.0, "parking_aisle": 8.0, "alley": 12.0,
    "drive-through": 8.0, "drive-thru": 8.0,
}
ACCESS_KEYS = ("motorcar", "motor_vehicle", "vehicle", "access")
PUBLIC_ACCESS = {"yes", "designated", "permissive", "official"}
BLOCKED_ACCESS = {
    "no", "private", "customers", "delivery", "agricultural", "forestry",
    "emergency", "permit", "residents", "dismount", "use_sidepath",
}
CLOSED_BARRIERS = {
    "bollard", "block", "bus_trap", "cycle_barrier", "chain", "debris", "ditch",
    "jersey_barrier", "kissing_gate", "log", "planter", "sally_port", "stile",
    "sump_buster", "tank_trap", "turnstile", "wall", "fence",
}


def access_value(tags, direction=None):
    suffix = ":forward" if direction == 1 else ":backward"
    for key in ACCESS_KEYS:
        if direction is not None and tags.get(key + suffix):
            return tags[key + suffix]
        if tags.get(key):
            return tags[key]
    return ""


def access_rule(tags, direction=None):
    value = access_value(tags, direction)
    if value in BLOCKED_ACCESS:
        return "blocked"
    if value == "destination":
        return "local"
    return "public"


def node_blocked(tags):
    value = access_value(tags)
    if value in PUBLIC_ACCESS or value == "destination":
        return False
    return value in BLOCKED_ACCESS or tags.get("barrier") in CLOSED_BARRIERS


def oneway_rule(tags):
    for key in ("oneway:motorcar", "oneway:motor_vehicle", "oneway:vehicle", "oneway"):
        value = tags.get(key)
        if value in {"yes", "true", "1"}:
            return 1
        if value == "-1":
            return -1
        if value in {"no", "false", "0"}:
            return 0
        if value in {"reversible", "alternating"}:
            return None
    return int(tags.get("junction") in {"roundabout", "circular"} or tags.get("highway") == "motorway")


def turn_rule(tags, members):
    """Parse unconditional via-node rules and count relevant unsupported cases."""
    issues = Counter()
    modes = ("motorcar", "motor_vehicle", "vehicle")
    kind = next((tags[f"restriction:{mode}"] for mode in modes if tags.get(f"restriction:{mode}")), None)
    generic = kind is None
    except_modes = {part.strip() for part in tags.get("except", "").split(";")}
    if generic and except_modes.intersection(modes):
        issues["unapplied_motorcar_exceptions"] += 1
        return [], issues
    if generic:
        kind = tags.get("restriction")
    if any(tags.get(key) for key in ("restriction:conditional", *(f"restriction:{m}:conditional" for m in modes))):
        issues["unsupported_conditional_turn_relations"] += 1
    if not kind:
        return [], issues
    if not kind.startswith(("no_", "only_")):
        issues["unsupported_turn_value"] += 1
        return [], issues

    via = [ref for member, ref, role in members if member == "n" and role == "via"]
    if any(member == "w" and role == "via" for member, _, role in members):
        issues["unsupported_via_way_relations"] += 1
        return [], issues
    approach = [ref for member, ref, role in members if member == "w" and role == "from"]
    departure = [ref for member, ref, role in members if member == "w" and role == "to"]
    if len(via) != 1 or not approach or not departure:
        issues["invalid_via_node_relations"] += 1
        return [], issues
    return [
        dict(via_node=via[0], from_way=from_way, to_way=to_way, kind=kind)
        for from_way in approach for to_way in departure
    ], issues


@dataclass(frozen=True)
class Route:
    edge_ids: tuple[int, ...]
    cost_s: float


class Graph:
    """Routing over source OSM node IDs without geometric merging of roads."""

    def __init__(self, edges, nodes, controls):
        if not np.array_equal(edges.edge_id.to_numpy(), np.arange(len(edges))):
            raise ValueError("edge_id must equal row position in the persisted edge table")
        self.edges = edges
        self.nodes = nodes
        self.controls = controls
        self._node_ids = nodes.node_id.to_numpy(dtype=np.int64)
        self._node_positions = {int(node): i for i, node in enumerate(self._node_ids)}
        self._xy = nodes[["lon", "lat"]].to_numpy()
        self._u = np.array([self._node_positions[node] for node in edges.u], dtype=np.int32)
        self._v = np.array([self._node_positions[node] for node in edges.v], dtype=np.int32)
        self._way = edges.way_id.to_numpy(dtype=np.int64)
        self._local_access = edges.local_access.to_numpy(dtype=bool)
        self._local_service = edges.local_service.to_numpy(dtype=bool)
        self._cost = edges.base_cost_s.to_numpy(dtype=float)
        self._blocked_nodes = {self._node_positions[c["node_id"]] for c in controls["nodes"] if c["node_blocked"]}
        self._turns = defaultdict(list)
        for t in controls["turns"]:
            self._turns[(self._node_positions[t["via_node"]], t["from_way"])].append((t["to_way"], t["kind"]))

        endpoints = (np.array(sorted(self._blocked_nodes), dtype=int)
                     if self._blocked_nodes else np.empty(0, dtype=int))
        public = ~(self._local_access | self._local_service)
        self._snap_trees = {}
        for public_only, allowed in ((True, public), (False, np.ones(len(edges), dtype=bool))):
            keep = np.unique(np.r_[self._u[allowed], self._v[allowed]])
            keep = np.setdiff1d(keep, endpoints, assume_unique=True)
            if len(keep):
                self._snap_trees[public_only] = cKDTree(self._unit_sphere(self._xy[keep])), keep
        self._networks = {}
        self._router_key = None
        self._router = None

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        return cls(
            pd.read_parquet(directory / "edges.parquet"),
            pd.read_parquet(directory / "nodes.parquet"),
            json.loads((directory / "controls.json").read_text()),
        )

    @staticmethod
    def _unit_sphere(xy):
        lon, lat = np.radians(xy).T
        return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))

    def snap(self, points, *, public_only=True):
        """Map one or more (longitude, latitude) coordinates to nearest routable OSM nodes."""
        raw = np.asarray(points, dtype=float)
        scalar = raw.ndim == 1
        if not len(raw):
            return np.empty(0, dtype=np.int64)
        tree, keep = self._snap_trees[public_only]
        _, idx = tree.query(self._unit_sphere(np.atleast_2d(raw)))
        ids = self._node_ids[keep[idx]]
        return int(ids[0]) if scalar else ids

    def _turn_allowed(self, incoming, outgoing, rules):
        for to_way, kind in rules:
            matches = self._way[outgoing] == to_way
            if matches and self._way[incoming] == to_way:
                uturn = self._v[outgoing] == self._u[incoming]
                matches = uturn if kind.endswith("u_turn") else not uturn
            if (kind.startswith("only_") and not matches) or (kind.startswith("no_") and matches):
                return False
        return True

    def _state_network(self, relax_service_local):
        local = self._local_access | (self._local_service & (not relax_service_local))
        n_edges = len(self.edges)
        local_edges = np.flatnonzero(local)
        public_endpoints = np.zeros(len(self.nodes), dtype=bool)
        public_endpoints[self._u[~local]] = True
        public_endpoints[self._v[~local]] = True
        # Each edge arrives in its normal state. Local edges additionally have a state
        # for routes that began on local roads and have not yet reached a public road.
        prefix = np.full(n_edges, -1, dtype=np.int32)
        prefix[local_edges] = np.arange(n_edges, n_edges + len(local_edges), dtype=np.int32)
        origins_offset = n_edges + len(local_edges)
        shape = (origins_offset + len(self.nodes),) * 2
        departures = defaultdict(list)
        arrivals = defaultdict(list)
        for e, (u, v) in enumerate(zip(self._u, self._v)):
            departures[int(u)].append(e)
            if local[e]:
                arrivals[int(v)].append(int(prefix[e]))
                if not public_endpoints[v]:
                    arrivals[int(v)].append(e)
            else:
                arrivals[int(v)].append(e)
        indices, data_edges = array("i"), array("i")
        offsets = [0]

        def transition(state, edge):
            indices.append(int(state))
            data_edges.append(int(edge))

        for e in range(n_edges):
            node = int(self._v[e])
            if node not in self._blocked_nodes:
                rules = self._turns.get((node, self._way[e]), ())
                for next_edge in departures.get(node, ()):
                    if (not local[e] or local[next_edge]) and self._turn_allowed(e, next_edge, rules):
                        transition(next_edge, next_edge)
            offsets.append(len(indices))
        for e in local_edges:
            node = int(self._v[e])
            if node not in self._blocked_nodes:
                rules = self._turns.get((node, self._way[e]), ())
                for next_edge in departures.get(node, ()):
                    if self._turn_allowed(e, next_edge, rules):
                        transition(prefix[next_edge] if local[next_edge] else next_edge, next_edge)
            offsets.append(len(indices))
        for node in range(len(self.nodes)):
            for e in departures.get(node, ()):
                transition(prefix[e] if local[e] and not public_endpoints[node] else e, e)
            offsets.append(len(indices))

        return (
            np.asarray(indices), np.asarray(data_edges), np.asarray(offsets, dtype=np.int32), shape,
            origins_offset, dict(arrivals), np.r_[np.arange(n_edges), local_edges],
        )

    def prepare(self, edge_cost_s=None, *, relax_service_local=False):
        """Build a reusable router for an immutable edge-cost array in edge_id order."""
        if (self._router_key is not None and self._router_key[0] is edge_cost_s
                and self._router_key[1] == relax_service_local):
            return self._router
        cost = self._cost if edge_cost_s is None else np.asarray(edge_cost_s, dtype=float)
        if cost.shape != (len(self.edges),) or not np.all(np.isfinite(cost) & (cost > 0)):
            raise ValueError("Edge costs must be positive, finite and in edge_id order")
        if relax_service_local not in self._networks:
            self._networks[relax_service_local] = self._state_network(relax_service_local)
        network = self._networks[relax_service_local]
        indices, data_edges, offsets, shape, origin_offset, arrivals, state_edges = network
        csr = csr_matrix((cost[data_edges], indices, offsets), shape=shape)
        self._router_key = (edge_cost_s, relax_service_local)
        self._router = Router(csr, self._node_positions, origin_offset, arrivals, state_edges)
        return self._router

    def shortest_path(self, source, target, edge_cost_s=None, *, relax_service_local=False):
        return self.shortest_paths(source, [target], edge_cost_s, relax_service_local=relax_service_local)[int(target)]

    def shortest_paths(self, source, targets, edge_cost_s=None, *, relax_service_local=False):
        return self.prepare(edge_cost_s, relax_service_local=relax_service_local).shortest_paths(source, targets)


@dataclass(frozen=True)
class Router:
    csr: csr_matrix
    node_positions: dict
    origins_offset: int
    arrivals: dict
    state_edges: np.ndarray

    def shortest_paths(self, source: int, targets: Iterable[int]) -> dict[int, Route | None]:
        """Route once from source and decode each requested target, preserving all OSM edge IDs."""
        targets = [int(target) for target in targets]
        result = dict.fromkeys(targets)
        if not targets or source not in self.node_positions:
            return result
        if source in result:
            result[source] = Route((), 0.0)
        if set(targets) == {source}:
            return result
        origin = self.origins_offset + self.node_positions[source]
        distances, predecessors = dijkstra(self.csr, indices=origin, directed=True, return_predecessors=True)
        for target in targets:
            if target == source or target not in self.node_positions:
                continue
            candidates = self.arrivals.get(self.node_positions[target])
            if not candidates:
                continue
            end = min(candidates, key=distances.__getitem__)
            if not np.isfinite(distances[end]):
                continue
            chosen = []
            state = end
            while state != origin:
                chosen.append(int(self.state_edges[state]))
                state = predecessors[state]
            result[target] = Route(tuple(reversed(chosen)), float(distances[end]))
        return result
