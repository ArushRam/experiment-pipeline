import pandas as pd
import torch
from typing import List, Optional

nucleotide_map = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
}

def one_hot_encode_sequence(sequence: str) -> List[List[int]]:
    return [nucleotide_map[nucleotide] for nucleotide in sequence]

class SequenceDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for DNA sequences.
    """
    def __init__(
        self,
        data_path: Optional[str],
        data_df: Optional[pd.DataFrame],
        sequence_column: str,
        target_column: str,
        upstream_sequence: str = "",
        downstream_sequence: str = "",
    ):
        # Ensure that only one of data_path and data_df is provided
        assert data_path is not None or data_df is not None
        assert data_path is None or data_df is None

        if data_path is not None:
            self.data = pd.read_csv(data_path)
        else:
            self.data = data_df

        sequence_encodings = (
            self.data[sequence_column]
            .apply(lambda x: f"{upstream_sequence}{x}{downstream_sequence}")
            .apply(one_hot_encode_sequence)
        )
        self.sequences = torch.tensor(
            sequence_encodings.tolist(), dtype=torch.float32
        ).permute(0, 2, 1)

        self.labels = torch.tensor(
            self.data[target_column].to_list(), dtype=torch.float32
        )

        # Ensure that the number of sequences and labels match
        assert len(self.sequences) == len(self.labels)

        # Ensue that sequences have correct channel sizes
        assert self.sequences.size(1) == 4

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]