#!/usr/bin/env python3

"""Update a URDF in-place using structure and values from a YAML tree."""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from scipy.spatial.transform import Rotation as R


# ---------------- XML helpers ----------------
def _lname(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _find_child(elem: ET.Element, name: str) -> Optional[ET.Element]:
    for child in elem:
        if _lname(child.tag) == name:
            return child
    return None


def _find_children(elem: ET.Element, name: str) -> List[ET.Element]:
    return [child for child in elem if _lname(child.tag) == name]


# ---------------- Math helpers ----------------
def _normalize(vec: Tuple[float, float, float]) -> Tuple[float, float, float]:
    length = math.sqrt(sum(component * component for component in vec))
    if length == 0.0:
        return vec
    return vec[0] / length, vec[1] / length, vec[2] / length


def _parse_floats(text: str) -> Optional[Tuple[float, float, float]]:
    parts = text.split()
    if len(parts) != 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def _format_float(value: float) -> str:
    return f"{value:.12g}"



# ---------------- URDF model ----------------
@dataclass
class LinkInfo:
    name: str
    element: ET.Element
    inertial: Optional[ET.Element]
    mass_element: Optional[ET.Element]
    inertia_element: Optional[ET.Element]
    mass: Optional[float]


@dataclass
class JointInfo:
    name: str
    element: ET.Element
    joint_type: str
    parent: str
    parent_element: Optional[ET.Element]
    child: str
    child_element: Optional[ET.Element]
    limit_element: Optional[ET.Element]
    origin_rotation: R
    axis: Optional[Tuple[float, float, float]]
    axis_element: Optional[ET.Element]


class URDFModel:
    def __init__(self, root: ET.Element) -> None:
        self.root = root
        self.links: Dict[str, LinkInfo] = {}
        self.joints: Dict[str, JointInfo] = {}
        self._parse()

    def _parse(self) -> None:
        for link in _find_children(self.root, "link"):
            name = link.attrib.get("name", "")
            inertial = _find_child(link, "inertial")
            mass_element = _find_child(inertial, "mass") if inertial is not None else None
            inertia_element = _find_child(inertial, "inertia") if inertial is not None else None

            mass_value: Optional[float] = None
            if mass_element is not None:
                value_attr = mass_element.attrib.get("value")
                if value_attr is not None:
                    try:
                        mass_value = float(value_attr)
                    except ValueError:
                        mass_value = None

            self.links[name] = LinkInfo(
                name=name,
                element=link,
                inertial=inertial,
                mass_element=mass_element,
                inertia_element=inertia_element,
                mass=mass_value,
            )

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

            origin_rot = R.identity()
            if origin_el is not None:
                rpy_attr = origin_el.attrib.get("rpy")
                if rpy_attr is not None:
                    parsed_rpy = _parse_floats(rpy_attr)
                    if parsed_rpy is not None:
                        origin_rot = R.from_euler("XYZ", parsed_rpy, degrees=False)

            axis_vec: Optional[Tuple[float, float, float]] = None
            if axis_el is not None:
                axis_attr = axis_el.attrib.get("xyz")
                if axis_attr is not None:
                    parsed_axis = _parse_floats(axis_attr)
                    if parsed_axis is not None:
                        axis_vec = _normalize(parsed_axis)
            if axis_vec is None and joint_type in {"revolute", "continuous", "prismatic"}:
                axis_vec = (1.0, 0.0, 0.0)

            self.joints[joint_name] = JointInfo(
                name=joint_name,
                element=joint,
                joint_type=joint_type,
                parent=parent_name,
                parent_element=parent_el,
                child=child_name,
                child_element=child_el,
                limit_element=limit_el,
                origin_rotation=origin_rot,
                axis=axis_vec,
                axis_element=axis_el,
            )

    def roots(self) -> List[str]:
        all_links = set(self.links.keys())
        child_links = {info.child for info in self.joints.values() if info.child}
        roots = sorted(all_links - child_links)
        return roots or sorted(all_links)


# ---------------- Tree building ----------------
def build_tree(model: URDFModel) -> Dict[str, Any]:
    graph: Dict[str, List[Tuple[str, JointInfo]]] = {}
    for joint in model.joints.values():
        if not joint.parent or not joint.child:
            continue
        graph.setdefault(joint.parent, []).append((joint.child, joint))
    for children in graph.values():
        children.sort(key=lambda item: item[0])

    def node_for(
        link_name: str,
        incoming_joint: Optional[JointInfo] = None,
        current_rotation: Optional[R] = None,
    ) -> Dict[str, Any]:
        rotation = current_rotation if current_rotation is not None else R.identity()

        node: Dict[str, Any] = {}

        if incoming_joint is not None:
            via_info: Dict[str, Any] = {"joint": incoming_joint.name}
            if incoming_joint.joint_type != "fixed":
                limit = incoming_joint.limit_element.attrib if incoming_joint.limit_element is not None else {}
                via_info["limit"] = {
                    "lower": float(limit.get("lower")) if "lower" in limit else None,
                    "upper": float(limit.get("upper")) if "upper" in limit else None,
                    "effort": float(limit.get("effort")) if "effort" in limit else None,
                    "velocity": float(limit.get("velocity")) if "velocity" in limit else None,
                }
                if incoming_joint.axis is not None:
                    world_axis = rotation.apply(incoming_joint.axis)
                    via_info["axis_world"] = list(world_axis)
            node["via"] = via_info

        link_info = model.links.get(link_name)
        node["mass"] = float(link_info.mass) if link_info and link_info.mass is not None else None

        children = graph.get(link_name, [])
        if children:
            child_map: Dict[str, Any] = {}
            for child_name, joint in children:
                rotation_to_child = rotation * joint.origin_rotation
                child_map[child_name] = node_for(child_name, incoming_joint=joint, current_rotation=rotation_to_child)
            node["children"] = child_map

        return node

    tree: Dict[str, Any] = {}
    for root_name in model.roots():
        tree[root_name] = node_for(root_name, None, R.identity())
    return tree


# ---------------- YAML helpers ----------------
def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------- Comparison ----------------
class TreeMismatch(Exception):
    pass


def compare_trees(urdf_tree: Dict[str, Any], yaml_tree: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    link_map: Dict[str, Dict[str, Any]] = {}
    joint_map: Dict[str, Dict[str, Any]] = {}

    urdf_roots = list(urdf_tree.keys())
    yaml_roots = list(yaml_tree.keys())
    if len(urdf_roots) != len(yaml_roots):
        raise TreeMismatch(f"Root count differs: URDF {len(urdf_roots)} vs YAML {len(yaml_roots)}")

    if len(urdf_roots) > 1:
        if set(urdf_roots) != set(yaml_roots):
            raise TreeMismatch("Root names diverge when multiple roots are present")
        pairs: Iterable[Tuple[str, str]] = ((name, name) for name in sorted(urdf_roots))
    else:
        urdf_root = urdf_roots[0]
        yaml_root = yaml_roots[0]
        if urdf_root != yaml_root:
            warnings.append(f"Root link renamed: '{urdf_root}' -> '{yaml_root}'")
        pairs = ((urdf_root, yaml_root),)

    def walk(urdf_name: str, yaml_name: str, urdf_node: Dict[str, Any], yaml_node: Dict[str, Any], path: str) -> None:
        link_map[urdf_name] = {"yaml_name": yaml_name, "yaml_node": yaml_node, "path": path}

        urdf_children = urdf_node.get("children", {}) or {}
        yaml_children = yaml_node.get("children", {}) or {}

        if len(urdf_children) != len(yaml_children):
            raise TreeMismatch(f"Child count mismatch at {path}: URDF {len(urdf_children)} vs YAML {len(yaml_children)}")

        if not urdf_children:
            return

        if len(urdf_children) > 1:
            urdf_keys = set(urdf_children.keys())
            yaml_keys = set(yaml_children.keys())
            if urdf_keys != yaml_keys:
                raise TreeMismatch(f"Siblings mismatch under {path}: {sorted(urdf_keys)} vs {sorted(yaml_keys)}")
            ordered_children = sorted(urdf_children.keys())
            for child_name in ordered_children:
                child_path = f"{path}/{child_name}"
                walk(
                    child_name,
                    child_name,
                    urdf_children[child_name],
                    yaml_children[child_name],
                    child_path,
                )
        else:
            (urdf_child_name, urdf_child_node), = urdf_children.items()
            (yaml_child_name, yaml_child_node), = yaml_children.items()
            if urdf_child_name != yaml_child_name:
                warnings.append(
                    f"Link '{path}' child renamed: '{urdf_child_name}' -> '{yaml_child_name}'"
                )
            walk(
                urdf_child_name,
                yaml_child_name,
                urdf_child_node,
                yaml_child_node,
                f"{path}/{urdf_child_name}"
            )

        for child_name, urdf_child_node in urdf_children.items():
            yaml_child_name = child_name if len(urdf_children) > 1 else next(iter(yaml_children.keys()))
            yaml_child_node = yaml_children[yaml_child_name]
            urdf_via = urdf_child_node.get("via")
            yaml_via = yaml_child_node.get("via")
            if urdf_via is None or yaml_via is None:
                continue
            joint_name = urdf_via.get("joint")
            yaml_joint_name = yaml_via.get("joint")
            if joint_name != yaml_joint_name:
                raise TreeMismatch(f"Joint mismatch under {path}: '{joint_name}' vs '{yaml_joint_name}'")
            joint_map[joint_name] = {"yaml_via": yaml_via, "path": f"{path}->{joint_name}"}

    for urdf_root, yaml_root in pairs:
        walk(
            urdf_root,
            yaml_root,
            urdf_tree[urdf_root],
            yaml_tree[yaml_root],
            urdf_root,
        )

    return link_map, joint_map, warnings


# ---------------- Joint kinematics ----------------
def compute_joint_world_axes(model: URDFModel) -> Dict[str, Optional[Tuple[float, float, float]]]:
    graph: Dict[str, List[JointInfo]] = defaultdict(list)
    for joint in model.joints.values():
        if joint.parent and joint.child:
            graph[joint.parent].append(joint)

    rotations: Dict[str, R] = {}
    joint_axes: Dict[str, Optional[Tuple[float, float, float]]] = {}

    queue: deque[str] = deque(model.roots())
    for root in queue:
        rotations[root] = R.identity()

    visited_links: set[str] = set()
    while queue:
        parent = queue.popleft()
        if parent in visited_links:
            continue
        visited_links.add(parent)
        rotation_parent = rotations.get(parent, R.identity())
        for joint in graph.get(parent, []):
            rotation_to_child = rotation_parent * joint.origin_rotation
            if joint.axis is not None:
                world_axis = rotation_to_child.apply(joint.axis)
                joint_axes[joint.name] = tuple(world_axis)
            else:
                joint_axes[joint.name] = None
            if joint.child not in rotations:
                rotations[joint.child] = rotation_to_child
                queue.append(joint.child)

    return joint_axes


# ---------------- Updates ----------------
def update_link_properties(model: URDFModel, link_map: Dict[str, Dict[str, Any]]) -> List[str]:
    rename_logs: List[str] = []
    for original_name, payload in link_map.items():
        link_info = model.links.get(original_name)
        if link_info is None:
            continue
        yaml_name = payload["yaml_name"]
        yaml_node = payload["yaml_node"]

        if yaml_name != original_name:
            link_info.element.attrib["name"] = yaml_name
            link_info.name = yaml_name
            rename_logs.append(f"Link '{original_name}' renamed to '{yaml_name}'")

        mass_value = _to_float(yaml_node.get("mass"))
        if mass_value is not None and link_info.mass_element is not None:
            old_mass = link_info.mass if link_info.mass is not None else None
            link_info.mass_element.attrib["value"] = _format_float(mass_value)
            if old_mass is not None and old_mass > 0.0 and link_info.inertia_element is not None:
                scale = mass_value / old_mass
                for attr in ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]:
                    val = link_info.inertia_element.attrib.get(attr)
                    if val is None:
                        continue
                    try:
                        numerical = float(val)
                    except ValueError:
                        continue
                    link_info.inertia_element.attrib[attr] = _format_float(numerical * scale)
            link_info.mass = mass_value

        if yaml_name != original_name:
            model.links[yaml_name] = link_info
            del model.links[original_name]
    return rename_logs


def update_joint_properties(
    model: URDFModel,
    joint_map: Dict[str, Dict[str, Any]],
    link_map: Dict[str, Dict[str, Any]],
    joint_axes: Dict[str, Optional[Tuple[float, float, float]]],
) -> Tuple[List[str], List[str]]:
    rename_lookup = {old: payload["yaml_name"] for old, payload in link_map.items()}
    limit_logs: List[str] = []
    axis_logs: List[str] = []

    for joint_name, joint_info in model.joints.items():
        new_parent = rename_lookup.get(joint_info.parent, joint_info.parent)
        new_child = rename_lookup.get(joint_info.child, joint_info.child)
        if joint_info.parent_element is not None:
            joint_info.parent_element.attrib["link"] = new_parent
        if joint_info.child_element is not None:
            joint_info.child_element.attrib["link"] = new_child
        joint_info.parent = new_parent
        joint_info.child = new_child

        yaml_payload = joint_map.get(joint_name)
        if yaml_payload is not None:
            yaml_via = yaml_payload["yaml_via"]
            limit_data = yaml_via.get("limit") if isinstance(yaml_via, dict) else None
            if limit_data and joint_info.joint_type != "fixed":
                if joint_info.limit_element is None:
                    joint_info.limit_element = ET.SubElement(joint_info.element, "limit")
                for attr in ["lower", "upper", "effort", "velocity"]:
                    value = _to_float(limit_data.get(attr))
                    if value is None:
                        if attr in joint_info.limit_element.attrib:
                            del joint_info.limit_element.attrib[attr]
                        continue
                    joint_info.limit_element.attrib[attr] = _format_float(value)
                limit_logs.append(f"Joint '{joint_name}' limits updated")

            yaml_axis = yaml_via.get("axis_world") if isinstance(yaml_via, dict) else None
            if (
                yaml_axis is not None
                and joint_info.joint_type not in {"fixed"}
                and joint_info.axis is not None
            ):
                axis_world_yaml = tuple(_to_float(v) or 0.0 for v in yaml_axis)
                axis_world_urdf = joint_axes.get(joint_name)
                if axis_world_urdf is not None:
                    dot = sum(a * b for a, b in zip(axis_world_yaml, axis_world_urdf))
                    if dot < 0.0:
                        flipped_axis = tuple(-component for component in joint_info.axis)
                        if joint_info.axis_element is None:
                            joint_info.axis_element = ET.SubElement(joint_info.element, "axis")
                        joint_info.axis_element.attrib["xyz"] = " ".join(_format_float(c) for c in flipped_axis)
                        joint_info.axis = flipped_axis
                        axis_logs.append(f"Joint '{joint_name}' axis flipped to match YAML")

        # Remove optional tags not needed in output
        for child in list(joint_info.element):
            tag_name = _lname(child.tag)
            if tag_name in {"dynamics", "safety_controller"}:
                joint_info.element.remove(child)

    return limit_logs, axis_logs


# ---------------- Output helpers ----------------
def _format_node(node: ET.Element, level: int) -> List[str]:
    if node.tag is ET.Comment:
        indent = "  " * level
        comment_text = node.text or ""
        return [f"{indent}<!--{comment_text}-->"]
    return _format_element(node, level)


def _format_element(element: ET.Element, level: int) -> List[str]:
    indent = "  " * level
    tag = _lname(element.tag)

    lines: List[str] = [f"{indent}<{tag}"]
    attr_lines = [f"{indent}  {key}=\"{value}\"" for key, value in element.attrib.items()]

    children = list(element)
    text = (element.text or "").strip()
    has_children = bool(children)
    has_text = bool(text)

    if has_children or has_text:
        if attr_lines:
            attr_lines[-1] += ">"
            lines.extend(attr_lines)
        else:
            lines[-1] += ">"

        if has_text:
            lines.append(f"{indent}  {text}")

        for child in children:
            lines.extend(_format_node(child, level + 1))

        lines.append(f"{indent}</{tag}>")
    else:
        if attr_lines:
            attr_lines[-1] += " />"
            lines.extend(attr_lines)
        else:
            lines[-1] += " />"

    return lines


# ---------------- Main ----------------
def load_yaml_tree(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping")
    return data


def main(argv: List[str]) -> int:
    if len(argv) not in {3, 4}:
        print("Usage: python yaml_to_urdf.py input.urdf tree.yaml [output.urdf]", file=sys.stderr)
        return 2

    urdf_path = argv[1]
    yaml_path = argv[2]
    output_path = argv[3] if len(argv) == 4 else urdf_path

    root = ET.parse(urdf_path).getroot()
    model = URDFModel(root)
    urdf_tree = build_tree(model)
    yaml_tree = load_yaml_tree(yaml_path)

    try:
        link_map, joint_map, warnings = compare_trees(urdf_tree, yaml_tree)
    except TreeMismatch as exc:
        print(f"Tree mismatch: {exc}", file=sys.stderr)
        return 1

    for warning in warnings:
        print(f"[warning] {warning}")

    joint_axes = compute_joint_world_axes(model)

    rename_logs = update_link_properties(model, link_map)
    limit_logs, axis_logs = update_joint_properties(model, joint_map, link_map, joint_axes)

    for log in rename_logs:
        print(f"[info] {log}")
    for log in limit_logs:
        print(f"[info] {log}")
    for log in axis_logs:
        print(f"[info] {log}")

    lines = ["<?xml version=\"1.0\" encoding=\"utf-8\"?>"]
    lines.extend(_format_element(root, 0))
    lines.append("")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Updated URDF written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
