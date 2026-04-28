from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

import app
from modeling import FEATURE_COLUMNS, TARGET_COLUMN, find_nearest_patients, load_data


class GraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_data(Path("framingham.csv"))
        cls.texts = app.TRANSLATIONS["en"]
        cls.input_frame = pd.DataFrame([cls.data.iloc[0][FEATURE_COLUMNS].to_dict()])
        cls.nearest = find_nearest_patients(cls.input_frame, cls.data, top_k=50)
        cls.validated_reference, _ = app.validate_data(cls.data)

    def test_build_similarity_graph_data_has_expected_node_count(self) -> None:
        graph = app.build_similarity_graph_data(self.data, self.input_frame, self.nearest["neighbors"], self.texts)
        self.assertEqual(len(graph["nodes"]), len(self.validated_reference) + 1)
        self.assertEqual(graph["nodes"][0]["id"], "selected")

    def test_build_similarity_graph_data_has_unique_node_ids(self) -> None:
        graph = app.build_similarity_graph_data(self.data, self.input_frame, self.nearest["neighbors"], self.texts)
        node_ids = [node["id"] for node in graph["nodes"]]
        self.assertEqual(len(node_ids), len(set(node_ids)))

    def test_build_similarity_graph_data_contains_selected_edges(self) -> None:
        graph = app.build_similarity_graph_data(self.data, self.input_frame, self.nearest["neighbors"], self.texts)
        selected_edges = [edge for edge in graph["edges"] if edge["from"] == "selected" or edge["to"] == "selected"]
        self.assertEqual(len(selected_edges), 50)

    def test_build_similarity_graph_data_internal_edges_are_present(self) -> None:
        graph = app.build_similarity_graph_data(self.data, self.input_frame, self.nearest["neighbors"], self.texts)
        internal_edges = [edge for edge in graph["edges"] if edge["from"] != "selected" and edge["to"] != "selected"]
        self.assertGreater(len(internal_edges), 0)

    def test_neighbor_nodes_have_tooltips(self) -> None:
        graph = app.build_similarity_graph_data(self.data, self.input_frame, self.nearest["neighbors"], self.texts)
        neighbor_nodes = [node for node in graph["nodes"] if node["id"] != "selected"]
        self.assertTrue(all("title" in node and str(node["title"]).strip() for node in neighbor_nodes))
        self.assertTrue(any(str(self.nearest["neighbors"].iloc[0]["_source_row_id"]) in str(node["title"]) for node in neighbor_nodes))


if __name__ == "__main__":
    unittest.main()
