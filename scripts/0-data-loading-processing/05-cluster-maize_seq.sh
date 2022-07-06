#!/bin/bash

#    Script used to cluster the promoter sequences with up to 80% sequence identity
#    before creating train and test splits.
#    Inputs:
#        maize_nam.csv: csv file of promoter sequences corresponding to cultivar/gene combinations
#    Outputs:
#        clustered/: a directory with clustered promoter names indexed by a
#            representative promoter from each cluster.
#            Note: use clustered/maize_nam.csv_cluster.tsv file for creating train/test split.

if [ -z "$1" ]; then
    ROOT="$(python -c 'from florabert import config; print(config.data_processed)')"
    INPUT_FILE="${ROOT}/combined/maize_nam.csv"
else
    INPUT_FILE="$1"
fi

DIR=$(cd $(dirname ${INPUT_FILE}) && pwd)
NAME=$(basename ${INPUT_FILE})
echo "Working in ${DIR}/clustered..."
mkdir -p "${DIR}/clustered"
echo "Writing fasta..."
awk '(NR>1){print ">"$2"\n"$3}' FS=',' $1 > "${DIR}/clustered/${NAME}.fa"
echo "Finished writing fasta."
echo "Running clustering..."
docker run -it -v $(echo "${DIR}/clustered"):/output -w /output soedinglab/mmseqs2 \
	mmseqs easy-cluster "${NAME}.fa" "${NAME}" /tmp --min-seq-id 0.8
echo "Finished clustering."
awk '{a[$1]+=1} END {
    b=c=0;
    for(x in a){b+=a[x];c+=1}
    print "n_clusters: "length(a)", avg_cluster_size: "b/c}' "${DIR}/clustered/${NAME}_cluster.tsv"

