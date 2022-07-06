from florabert import config
import pandas as pd


if __name__ == '__main__':
    # Read the gene expression file, seperated by tab
    print("Reading gene expressiond data")
    df = pd.read_csv(config.data_raw/'gene_expression'/'TPM_expression_counts_from_B73.txt',sep = '\t')
    # write the gene expression file, seperated by tab to the processed folder
    df_tissue = df.drop(columns = ['Run','growth_condition','Cultivar','Developmental_stage','Age'])
    df_tissue_mean = df_tissue.groupby('organism_part',as_index = False).mean()
    df_tissue_mean.set_index('organism_part',inplace=True)
    gene_expression = df_tissue_mean.T
    print("Done.")
    print("Saving gene expression data.")
    
    gene_x_dir = config.data_final / 'Zmb73'
    if not gene_x_dir.exists():
        gene_x_dir.mkdir()
        
    gene_expression.to_csv(gene_x_dir / 'B73_genex.txt', sep="\t")
    
    print("Finished processing: gene expression")

    



    
