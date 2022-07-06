""" Script used to combine all the downloaded sequences within all the databases
    ['Ensembl', 'Refseq', 'Maize', 'Maize_addition'].
    INPUTS:
        - The processed sequences for each database located under
        config.data_processed
    OUTPUTS:
        - One dataframe per database with all the combined sequences
        within that database, stored inside config.data_processed / "combined"

    Execute "python3 03-combine-databases.py" to run.
"""
import os
import pandas as pd
from florabert import config
from Bio import SeqIO
from tqdm import tqdm


def combine_data(database, suffix):
    """Combine sequences for species within database into a csv"""
    parent_path = config.data_processed / database
    species_list = [c for c in os.listdir(parent_path) if "DS_Store" not in c]
    print(len(species_list))

    # Define the four columns of the df
    df_dict = {"species": [], "name": [], "sequence": [], "file_path": []}

    # Read in the content of each species and append to df one by one
    for species in tqdm(species_list):
        files = os.listdir(parent_path / species)
        try:
            fa_name = [f for f in files if f.endswith(suffix)][0]
        except:
            print(f"{species} has no processed data available.")
            continue
        file_path = parent_path / species / fa_name
        sequences = SeqIO.parse(open(file_path), "fasta")
        for seq in sequences:
            df_dict["species"].append(species)
            df_dict["name"].append(seq.name)
            df_dict["sequence"].append(str(seq.seq).upper())
            df_dict["file_path"].append(file_path)
    df = pd.DataFrame(df_dict)
    print(
        f"Finished processing {database} with {df['species'].nunique()} species, resulting in shape {df.shape}."
    )
    return df


def add_b73(df: pd.DataFrame) -> pd.DataFrame:
    """Add B73 to combined dataframe."""
    b73_dir = config.data_processed / "Maize" / "Zmb73"
    species = "zm-b73"
    file_path = list(b73_dir.glob("*.fa"))[0]
    # Define the four columns of the df
    df_dict = {"species": [], "name": [], "sequence": [], "file_path": []}

    sequences = SeqIO.parse(open(file_path), "fasta")
    for seq in sequences:
        df_dict["species"].append(species)
        df_dict["name"].append(seq.name)
        df_dict["sequence"].append(str(seq.seq).upper())
        df_dict["file_path"].append(file_path)

    df_b73 = pd.DataFrame(df_dict)
    return pd.concat((df_b73, df))


if __name__ == "__main__":
    save_path = config.data_processed / "combined"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Combine All the databases
    for database in ["Maize_nam"]:  # 'Ensembl', 'Refseq', 'Maize', 'Maize_addition',
        suffix = ".fna" if database == "Refseq" else ".fa"
        combined_df = combine_data(database, suffix)
        if database == "Maize_nam":
            combined_df = add_b73(combined_df)
        combined_df.to_csv(save_path / f"{database.lower()}.csv", index=None)
