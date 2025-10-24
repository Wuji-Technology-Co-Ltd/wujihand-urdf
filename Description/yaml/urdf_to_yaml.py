#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parse URDF -> tree (via -> mass -> children) -> YAML (using PyYAML).

Usage:
  python urdf_to_yaml.py input.urdf [output.yaml]
"""

import sys
import math
import xml.etree.ElementTree as ET
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, Optional, List, Tuple
import yaml
from yaml.dumper import SafeDumper
from yaml.nodes import ScalarNode, SequenceNode

from scipy.spatial.transform import Rotation as R


# ---------------- XML helpers ----------------
def _lname(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _find_child(elem: ET.Element, name: str) -> Optional[ET.Element]:
    for c in elem:
        if _lname(c.tag) == name:
            return c
    return None


def _find_children(elem: ET.Element, name: str) -> List[ET.Element]:
    return [c for c in elem if _lname(c.tag) == name]


# ---------------- YAML helpers ----------------
class FlowList(list):
    pass


def _represent_flow_list(dumper: SafeDumper, data: FlowList) -> SequenceNode:
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def _represent_decimal(dumper: SafeDumper, data: Decimal) -> ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:float", format(data))


SafeDumper.add_representer(FlowList, _represent_flow_list)
SafeDumper.add_representer(Decimal, _represent_decimal)


# ---------------- Math helpers ----------------
def _normalize(vec: Tuple[float, float, float]) -> Tuple[float, float, float]:
    length = math.sqrt(sum(c * c for c in vec))
    if length == 0.0:
        return vec
    return (
        vec[0] / length,
        vec[1] / length,
        vec[2] / length,
    )


def _parse_floats(text: str) -> Optional[Tuple[float, float, float]]:
    parts = text.split()
    if len(parts) != 3:
        return None
    try:
        return (
            float(parts[0]),
            float(parts[1]),
            float(parts[2]),
        )
    except ValueError:
        return None


def _format_axis_world(vec: Tuple[float, float, float]) -> FlowList:
    formatted = FlowList()
    for component in vec:
        dec = Decimal(str(component)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if dec == 0:
            dec = Decimal("0.00")
        formatted.append(dec)
    return formatted


# ---------------- URDF model ----------------
class URDFModel:
    def __init__(self, root: ET.Element) -> None:
        self.root = root
        self.links: Dict[str, Dict[str, Any]] = {}
        self.joints: Dict[str, Dict[str, Any]] = {}
        self._parse()

    def _parse(self) -> None:
        # links
        for link in _find_children(self.root, "link"):
            name = link.attrib.get("name", "")
            mass_val: Optional[float] = None
            inertial = _find_child(link, "inertial")
            if inertial is not None:
                mass = _find_child(inertial, "mass")
                if mass is not None:
                    v = mass.attrib.get("value")
                    if v is not None:
                        try:
                            mass_val = float(v)
                        except ValueError:
                            pass
            self.links[name] = {"name": name, "mass": mass_val}

        # joints
        for joint in _find_children(self.root, "joint"):
            joint_name = joint.attrib.get("name", "")
            joint_type = joint.attrib.get("type", "")
            parent_el = _find_child(joint, "parent")
            child_el = _find_child(joint, "child")
            limit_el = _find_child(joint, "limit")
            origin_el = _find_child(joint, "origin")
            axis_el = _find_child(joint, "axis")

            parent_name = parent_el.attrib.get("link") if parent_el is not None else ""
            child_name = child_el.attrib.get("link") if child_el is not None else ""

            lower = upper = effort = velocity = None
            if limit_el is not None:
                limit_attrib = limit_el.attrib

                def _parse_float(attr: str) -> Optional[float]:
                    if attr not in limit_attrib:
                        return None
                    try:
                        return float(limit_attrib[attr])
                    except ValueError:
                        return None

                lower = _parse_float("lower")
                upper = _parse_float("upper")
                effort = _parse_float("effort")
                velocity = _parse_float("velocity")

            origin_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
            origin_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
            origin_rot = R.identity()
            if origin_el is not None:
                xyz_attr = origin_el.attrib.get("xyz")
                if xyz_attr is not None:
                    parsed_xyz = _parse_floats(xyz_attr)
                    if parsed_xyz is not None:
                        origin_xyz = parsed_xyz
                rpy_attr = origin_el.attrib.get("rpy")
                if rpy_attr is not None:
                    parsed_rpy = _parse_floats(rpy_attr)
                    if parsed_rpy is not None:
                        origin_rpy = parsed_rpy
                        origin_rot = R.from_euler("XYZ", origin_rpy, degrees=False)

            axis_vec: Optional[Tuple[float, float, float]] = None
            if axis_el is not None:
                axis_attr = axis_el.attrib.get("xyz")
                if axis_attr is not None:
                    parsed_axis = _parse_floats(axis_attr)
                    if parsed_axis is not None:
                        axis_vec = _normalize(parsed_axis)
            if axis_vec is None and joint_type in {
                "revolute",
                "continuous",
                "prismatic",
            }:
                axis_vec = (1.0, 0.0, 0.0)

            self.joints[joint_name] = {
                "name": joint_name,
                "type": joint_type,
                "parent": parent_name,
                "child": child_name,
                "limit": {
                    "lower": lower,
                    "upper": upper,
                    "effort": effort,
                    "velocity": velocity,
                },
                "origin": {
                    "xyz": origin_xyz,
                    "rpy": origin_rpy,
                },
                "origin_rotation": origin_rot,
                "axis": axis_vec,
            }

    def roots(self) -> List[str]:
        all_links = set(self.links.keys())
        child_links = {j["child"] for j in self.joints.values() if j.get("child")}
        roots = sorted(all_links - child_links)
        return roots or sorted(all_links)


# ---------------- Build ordered tree ----------------
def build_tree(model: URDFModel) -> Dict[str, Any]:
    # parent_link -> list[(child_link, joint_dict)]
    graph: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for j in model.joints.values():
        p, c = j["parent"], j["child"]
        if not p or not c:
            continue
        graph.setdefault(p, []).append((c, j))
    for k in graph:
        graph[k].sort(key=lambda x: x[0])  # stable order

    def node_for(
        link_name: str,
        incoming_joint: Optional[Dict[str, Any]] = None,
        current_rotation: Optional[R] = None,
    ) -> Dict[str, Any]:
        if current_rotation is None:
            current_rotation = R.identity()

        node: Dict[str, Any] = {}

        if incoming_joint is not None:
            node["via"] = incoming_joint

        mass_val = model.links.get(link_name, {}).get("mass", None)
        node["mass"] = float(mass_val) if mass_val is not None else None

        children = graph.get(link_name, [])
        if children:
            child_map: Dict[str, Any] = {}
            for child_name, joint_data in children:
                rotation_to_child = current_rotation * joint_data["origin_rotation"]

                via_info: Dict[str, Any] = {"joint": joint_data["name"]}

                if joint_data.get("type") != "fixed":
                    via_info["limit"] = {
                        "lower": joint_data["limit"]["lower"],
                        "upper": joint_data["limit"]["upper"],
                        "effort": joint_data["limit"]["effort"],
                        "velocity": joint_data["limit"]["velocity"],
                    }

                    axis_vec = joint_data.get("axis")
                    if axis_vec is not None:
                        world_axis = rotation_to_child.apply(axis_vec)
                        via_info["axis_world"] = _format_axis_world(tuple(world_axis))

                child_map[child_name] = node_for(
                    child_name,
                    incoming_joint=via_info,
                    current_rotation=rotation_to_child,
                )
            node["children"] = child_map

        return node

    tree: Dict[str, Any] = {}
    for r in model.roots():
        tree[r] = node_for(r, incoming_joint=None, current_rotation=R.identity())
    return tree


# ---------------- CLI ----------------
def main(argv: List[str]) -> int:
    if len(argv) < 2 or len(argv) > 3:
        print("Usage: python urdf_to_yaml.py input.urdf [output.yaml]", file=sys.stderr)
        return 2

    in_path = argv[1]
    out_path = argv[2] if len(argv) == 3 else None

    root = ET.parse(in_path).getroot()
    model = URDFModel(root)
    struct = build_tree(model)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(struct, f, allow_unicode=True, sort_keys=False)
        print(f"Wrote YAML to {out_path}")
    else:
        print(yaml.safe_dump(struct, allow_unicode=True, sort_keys=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
