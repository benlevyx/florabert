""" Python script used to download and process gene sequences from a specified
    database: 'Ensembl', 'Refseq', 'Maize', 'Maize_addition', 'Maize_nam'.
    INPUTS:
        - Four <db_name>_link.csv that stores the links to gz files on 
        the ftp databases, located inside config.data_raw / "gz_link"
    OUTPUTS:
        - A single folder with the database name, located inside 
        config.data_processed, that stores the processed sequences
        for individual species / cultivars
    Sample usage: `python 02-download-process-db-data.py Ensembl` to download
    processed regulatory sequences from the Ensembl database
"""
import os
import argparse
import pandas as pd
from florabert import config
from florabert import gene_db_io


if __name__ == "__main__":

    # Parse db_name
    parser = argparse.ArgumentParser(description="Provide DB name to be processed")

    # Add the arguments
    parser.add_argument("DB", metavar="db", type=str, help="Name of database")

    # Execute parse_args() to get db_name
    args = parser.parse_args()
    db_name = args.DB[0].upper() + args.DB[1:].lower()
    if db_name not in ["Ensembl", "Refseq", "Maize", "Maize_addition", "Maize_nam"]:
        raise ValueError(
            "Arg db_name must be one of ['Ensembl', 'Refseq', 'Maize', 'Maize_addition', 'Maize_nam']."
        )
    print("Name of database to be processed:", db_name)

    # Read in the db_link.csv
    df = pd.read_csv(
        os.path.join(config.data_raw, "gz_link", f"{db_name.lower()}_link.csv")
    )

    # Loop through the rows of link file
    for idx, row in df.iterrows():
        dna_url = row["gene_link"]
        annot_url = row["annot_link"]
        dna_name = row["gene_link"].split("/")[-1].replace(".gz", "")
        annot_name = row["annot_link"].split("/")[-1].replace(".gz", "")
        species_name = row["name"].lower()

        # Create paths to be added
        db_path = config.data_raw / db_name
        dna_path = db_path / "dna"
        annot_path = db_path / "annot"
        processed_db_path = config.data_processed / db_name

        # Test run the main function
        try:
            gene_db_io.generate_sequence_for_species(
                dna_name,
                annot_name,
                dna_url,
                annot_url,
                dna_path,
                annot_path,
                processed_db_path,
                db_name,
                species_name,
                regulatory_len=1000,
            )

            print(f"\nFinished processing index {idx}:", dna_name)

            # Test load process dna sequences
            processed_fa = gene_db_io.load_processed_fa(
                processed_db_path, dna_name, db_name, species_name
            )
            for seq in processed_fa:
                print("First Sequence in the processed file:")
                print(seq.id, "\n", seq.seq)
                break
        except (KeyError, TypeError, ValueError, IndexError) as e:
            print(f"\nUnable to process {dna_name} with Exception {e}.")
