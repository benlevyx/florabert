
#!/bin/bash 

# get sra-toolkit OR this should be included within docker using command line 
# use docker rnaseq image as described here: https://github.com/inari-ag/Docker/tree/master/inari-rnaseq-jupyter
# in docker set up required directories if not already present:
## mkdir /home/ubuntu/align_against_NAM_genomes
## mkdir /home/ubuntu/align_against_NAM_genomes/read_count
## mkdir /home/ubuntu/align_against_NAM_genomes/indices

# outside of docker copy over reference genome index and gtf annotation
#copy over the reference, reference index, and annotation (or use the script)

#Create a list of samples called 'full_sample_list' formatted as follows with names of the ref gens nd indices
#ERR3791403	B97	Zm-B97-REFERENCE-NAM-1.0	Zm00018ab.1
#ERR3791404	B97	Zm-B97-REFERENCE-NAM-1.0	Zm00018ab.1
#ERR3791405	CML103	Zm-CML103-REFERENCE-NAM-1.0	Zm00021ab.1
#ERR3791406	CML103	Zm-CML103-REFERENCE-NAM-1.0	Zm00021ab.1

#split that list as follows:
# split -n l/12 full_sample_list sample_list

##run this as:
# bash /home/ubuntu/align_against_NAM_genomes/alignment_script.sh /home/ubuntu/align_against_NAM_genomes/sample_list#

dir="/home/ubuntu/align_against_NAM_genomes/read_count"
index_dir="/home/ubuntu/align_against_NAM_genomes/indices"
anno_dir="/home/ubuntu/align_against_NAM_genomes/ref_files"

#file="/home/ubuntu/align_against_NAM_genomes/sample_list1"
file=$1

while read -r smpname cultivar refgen annotation; do
	i="$smpname"
	echo "$i"
	# get data 
	sleep 1.25 
	/home/ubuntu/test/sratoolkit.2.10.8-ubuntu64/bin/fasterq-dump -e 3 --split-files ${i} -O ${dir} && 
	#fastq-dump --split-files ${i} -O ${dir} &&

	# trimming 
	trimmomatic PE \
		$dir/${i}_1.fastq $dir/${i}_2.fastq \
		$dir/${i}_paired_1.fq.gz \
		$dir/${i}_unpaired_1.fq.gz \
		$dir/${i}_paired_2.fq.gz \
		$dir/${i}_unpaired_2.fq.gz \
		ILLUMINACLIP::2:30:10 \
		LEADING:3 TRAILING:3 \
		SLIDINGWINDOW:4:15 \
		MINLEN:36 \
		-threads 22 && >> ${dir}/${i}.trim.log
	
	rm $dir/${i}_1.fastq
	rm $dir/${i}_2.fastq

	# mapping 	
	hisat2 -x ${index_dir}/${refgen}_${annotation} \
		-1 ${dir}/${i}_paired_1.fq.gz \
		-2 ${dir}/${i}_paired_2.fq.gz \
		--met-file ${dir}/${i}.err \
		--summary-file ${dir}/${i}.out \
		--no-unal \
		-p 28 | samtools view -Sb | samtools sort -o ${dir}/${i}.bam -@ 28 &&
	
	rm $dir/${i}_paired_1.fq.gz
	rm $dir/${i}_unpaired_1.fq.gz
	rm $dir/${i}_paired_2.fq.gz
	rm $dir/${i}_unpaired_2.fq.gz 

	# extract feature count 
	featureCounts -T 28 --primary -C -t exon -g gene_id \
		-a ${anno_dir}/${refgen}_${annotation}.gtf \
		-o ${dir}/${i}.count.txt \
		${dir}/${i}.bam &>> ${dir}/${i}.counts.log &&
	
	# rm ${dir}/${i}.bam

	# generate read count file 
	echo Extracting read count from count.txt file in ${i}
	awk '{print $1, $7}' ${dir}/${i}.count.txt | sed 1d > ${dir}/${i}.abundance
done < "$file"
