from abc import abstractmethod
from typing import List, Optional

from llama_index import QueryBundle
from llama_index.schema import NodeWithScore





class DuplicateRemoverNodePostprocessor:
    """Node postprocessor."""

    @abstractmethod
    def postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        print("postprocess_node enter")

        unique_hashes = set()
        unique_node = []

        for node in nodes:
            node_hash = node.node,hash

            if node.hash not in  unique_hashes:
                unique_hashes.add(node_hash)
                unique_node.append(node)

        return unique_node