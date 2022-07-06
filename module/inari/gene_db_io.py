""" Scripts with helper functions to download and process gene sequences
    using raw DNA sequences and annotation files.
"""
import os
from Bio import SeqIO
from . import config, utils


# Define Global Vars
ANNOT_SUFFIX_DICT = {'Ensembl': 'gff3',
                     'Refseq': 'gff',
                     'Maize': 'gff3',
                     'Maize_addition': 'gff3',
                     'Maize_nam': 'gff3'}
GENE_SUFFIX_DICT = {'Ensembl': 'fa',
                    'Refseq': 'fna',
                    'Maize': 'fa',
                    'Maize_addition': 'fa',
                     'Maize_nam': 'fa'}


# Helper functions for download data from the selected Database
def generate_directories(db_name):
    """ Creates a list of directories to be added
        Params:
            db_name: str, name of the database to be processed
    """
    # Create paths to be added
    db_path = config.data_raw / db_name
    dna_path = db_path / 'dna'
    annot_path = db_path / 'annot'
    processed_db_path = config.data_processed / db_name

    for path in [db_path, dna_path, annot_path, processed_db_path]:
        if not os.path.exists(path):
            os.mkdir(path)


def faidx(dna_path, dna_name):
    """ Adapted from Inari noteobok. Used to extract .fai from .fa file
        Params:
            dna_path: str, directory for the .fa/.fna dna file
            dna_name: str, name for dna .fa/.fna file
    """
    exe_str = f"{config.samtools} faidx {os.path.join(dna_path, dna_name)}"
    utils.execute(exe_str)


def extract_flanking_region(annot_name, annot_path, db_name):
    """ Adapted from Inari noteobok. Used to extract the flanking region.
        Generates a .gene.gff3/gff file using the .gff3/gff file.
        Params:
            annot_name: str, name of the annotation .gff3 file
            annot_path: str, directory that stores the gff3 file
    """
    # Generate input and output file names
    in_path = os.path.join(annot_path, annot_name)
    suffix = ANNOT_SUFFIX_DICT[db_name]
    out_path = in_path.replace(f".{suffix}", f".gene.{suffix}")
    exe_str = f"grep -P '\tgene\t' {in_path} > {out_path}" # -P for ubuntu, -p for linux
    utils.execute(exe_str)


def get_1kbup(dna_name, annot_name, dna_path, annot_path, regulatory_len, db_name):
    """ Adapted from Inari noteobok. Creates the .gene.1kbup.gff3 file
        using the .gene.gff3 generated using extract_flanking_region.
        Params:
            dna_name: str, name for dna .fa/fna file
            annot_name: str, name of the annotation .gff3/gff file
            dna_path: str, directory for the .fa/fna dna file
            annot_path: str, directory that stores the gff3/gff file
            regulatory_len: int, length of regulatory region to be extracted
            db_name: str, name of the database to be processed
    """
    # Generate input and output file names
    suffix = ANNOT_SUFFIX_DICT[db_name]
    annot_gene = annot_name.replace(f".{suffix}", f".gene.{suffix}")
    annot_1kbup = annot_name.replace(f".{suffix}", f".gene.1kbup.{suffix}")

    # Create the exe command to get 1kbup
    exe_str = f"{config.bedtools} flank -i "
    exe_str += f"{os.path.join(annot_path, annot_gene)} "
    exe_str += f"-g {os.path.join(dna_path, dna_name)}.fai "
    exe_str += f"-l {str(regulatory_len)} "
    exe_str += "-r 0 "
    exe_str += "-s "
    exe_str += f"> {os.path.join(annot_path, annot_1kbup)}"
    utils.execute(exe_str)


def subtract(dna_name, annot_name, annot_path, db_name):
    ''' Adapted from Inari noteobok. Apply Bedtools subtract to subtract genic
        regions of neighbouring genes from the intergenic flanks.
        Params:
            dna_name: str, name for dna .fa/fna file
            annot_name: str, directory that stores the gff3/gff file
            annot_path: str, directory that stores the gff3/gff file
            db_name: str, name of the database to be processed
    '''
    # Generate input and output file names
    suffix = ANNOT_SUFFIX_DICT[db_name]
    annot_gene = annot_name.replace(f".{suffix}", f".gene.{suffix}")
    annot_1kbup = annot_name.replace(f".{suffix}", f".gene.1kbup.{suffix}")
    annot_nov = annot_name.replace(f".{suffix}", f".gene.1kbup.nov.{suffix}")

    exe_str = f"{config.bedtools} subtract -a " + \
              os.path.join(annot_path, annot_1kbup)
    exe_str += " -b " + os.path.join(annot_path, annot_gene)
    exe_str += " > " + os.path.join(annot_path, annot_nov)
    utils.execute(exe_str)


def remove_split_fragments(annot_name, annot_path, db_name):
    ''' Go over the gtf file and if multiple promoter fragments for one gene:
        retain only last one (largest coordinates) in case of positive strand
        retain only first one (smallest coordinates) in case of negative strand
        Params:
            annot_name:  str, name of the annotation .gff3 file
            annot_path: str, directory that stores the gff3 file
            db_name: str, name of the database to be processed
    '''
    suffix = ANNOT_SUFFIX_DICT[db_name]
    annot_nov = annot_name.replace(f".{suffix}", f".gene.1kbup.nov.{suffix}")
    annot_final = annot_name.replace(f".{suffix}", f".gene.1kbup.nov.final.{suffix}")

    # Read in all the extracted regions
    with open(os.path.join(annot_path, annot_nov)) as gtf_fh_in:
        fragment_dict = {}
        orientation_dict = {}
        for line in gtf_fh_in:
            line = line.rstrip()
            line_list = line.split('\t')
            if db_name == 'Ensembl':
                gene_id_temp = line_list[-1].split(';')[0]
                gene_id = gene_id_temp.split(":")[1]
            elif db_name == 'Refseq':
                gene_id_temp = line_list[-1].split(';')[1]
                gene_id = gene_id_temp.split(":")[1]
            elif db_name == 'Maize':
                gene_id_temp = line_list[-1].split(';')[0]
                gene_id = gene_id_temp.split(':')[1]
            else:
                gene_id_temp = line_list[-1].split(';')[0]
                gene_id = gene_id_temp.split('=')[1]
            orientation = line_list[6]
            line_list[2] = gene_id
            orientation_dict[gene_id] = orientation
            if gene_id not in fragment_dict:
                fragment_dict[gene_id] = []
            fragment_dict[gene_id].append("\t".join(line_list))

    # Write out only the retained ones
    with open(os.path.join(annot_path, annot_final), "w") as gtf_fh_out:
        for gene_id in fragment_dict:
            if orientation_dict[gene_id] == '+':
                # take fragment with highest coords, which is latest one added
                gtf_fh_out.write("{}\n".format(fragment_dict[gene_id][-1]))
            else:
                # orientation == -
                # take fragment with lowest coords, which is first one added
                gtf_fh_out.write("{}\n".format(fragment_dict[gene_id][0]))


def extract_sequence(dna_name, annot_name, dna_path, annot_path, save_path, db_name):
    ''' Adapted from Inari noteobok. Uses Bedtools getfasta to extract
        the final sequences, w/ extension ".gene.1kbup.nov.final.fa".
        Params:
            dna_name: str, name for dna .fa file
            annot_name: str, name of the annotation .gff3 file
            dna_path: str, directory for the .fa dna file
            annot_path: str, directory that stores the gff3 file
            save_path: str, directory for the output .fa file
            db_name: str, name of the database to be processed
    '''
    # Generate input and output file names
    suffix = ANNOT_SUFFIX_DICT[db_name]
    suffix_g = GENE_SUFFIX_DICT[db_name]
    annot_final = annot_name.replace(f".{suffix}", f".gene.1kbup.nov.final.{suffix}")
    dna_final = dna_name.replace(f".{suffix_g}", f".gene.1kbup.nov.final.{suffix_g}")

    exe_str = f"{config.bedtools} getfasta -fi " + \
              os.path.join(dna_path, dna_name)
    exe_str += " -bed " + os.path.join(annot_path, annot_final)
    exe_str += " -s -name+ > "
    exe_str += os.path.join(save_path, dna_final)
    utils.execute(exe_str)


def generate_sequence_for_species(dna_name, annot_name, dna_url, annot_url,
                                  dna_path, annot_path, save_path, db_name,
                                  species_name, regulatory_len=1000):
    """ The main function that chains the above methods together to extract the
        relevant sequesnces from the initial .fa/fna dna file and .gff3/gff
        annotation file downloaded from the Ensembl database.
        Params:
            dna_name: str, name for dna .fa/fna file
            annot_name: str, name of the annotation .gff3/gff file
            dna_url: str, url to download the .fa/fna file
            annot_url: str, url to download the .gff3/gff file
            dna_path: str, directory for the .fa/fna dna file
            annot_path: str, directory that stores the gff3/gff file
            save_path: str, directory for the output .fa/fna file
            db_name: str, name of the database to be processed
            species_name: str, name of species
            regulatory_len: int, length of regulatory region to be extracted
    """
    # Generate directories
    generate_directories(db_name)

    # Create species-specific directory
    species_name = species_name.strip().lower()
    dna_path_s = dna_path / species_name
    annot_path_s = annot_path / species_name
    save_path = save_path / species_name
    for path in [dna_path_s, annot_path_s, save_path]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Download the raw fa and gff files
    utils.download(dna_url, dna_path_s, db_name, dna_name)
    utils.download(annot_url, annot_path_s, db_name, annot_name)

    # Unzip the gz files
    utils.unzip(dna_path_s, dna_name)
    utils.unzip(annot_path_s, annot_name)

    # Run Faidx command
    faidx(dna_path_s, dna_name)

    # Only extract the flanking regions for the genes
    extract_flanking_region(annot_name, annot_path_s, db_name)

    # Get 1kbup file
    get_1kbup(dna_name, annot_name, dna_path_s, annot_path_s, regulatory_len, db_name)

    # Bedtools to subtract genic regions of neighbouring genes
    # from the intergenic flanks
    subtract(dna_name, annot_name, annot_path_s, db_name)

    # Remove Split fragments
    remove_split_fragments(annot_name, annot_path_s, db_name)

    # Extract the resulting sequence
    extract_sequence(dna_name, annot_name, dna_path_s, annot_path_s, save_path, db_name)

    # Remove raw sequence files
    utils.clear_folder(dna_path_s, to_continue='y')
    # utils.clear_folder(annot_path_s, to_continue='y')


def load_processed_fa(processed_db_path, dna_name, db_name, species_name):
    """ Helper function to load sequences with name dna_name.
        Params:
            processed_db_path: db_path within processed data folder
            dna_name: name of the .fa/fna file to be loaded
            db_name: str, name of the database to be processed
            species_name: str, name of species
    """
    suffix_g = GENE_SUFFIX_DICT[db_name]
    dna_final = dna_name.replace(f".{suffix_g}", f".gene.1kbup.nov.final.{suffix_g}")
    f_path = os.path.join(processed_db_path, species_name, dna_final)
    fasta_sequences = SeqIO.parse(open(f_path), 'fasta')
    return fasta_sequences
