#!/bin/bash

wget https://data.nextstrain.org/files/ncov/open/metadata.tsv.zst

zstd -d metadata.tsv.zst

rm metadata.tsv.zst

date=$(awk -F'\t' 'NR==1{for (i=1; i<=NF; i++) if ($i == "date") print i}' metadata.tsv)
region=$(awk -F'\t' 'NR==1{for (i=1; i<=NF; i++) if ($i == "region") print i}' metadata.tsv)
host=$(awk -F'\t' 'NR==1{for (i=1; i<=NF; i++) if ($i == "host") print i}' metadata.tsv)
clade=$(awk -F'\t' 'NR==1{for (i=1; i<=NF; i++) if ($i == "Nextstrain_clade") print i}' metadata.tsv)
lineage=$(awk -F'\t' 'NR==1{for (i=1; i<=NF; i++) if ($i == "pango_lineage") print i}' metadata.tsv)
deletions=$(awk -F'\t' 'NR==1{for (i=1; i<=NF; i++) if ($i == "deletions") print i}' metadata.tsv)
insertions=$(awk -F'\t' 'NR==1{for (i=1; i<=NF; i++) if ($i == "insertions") print i}' metadata.tsv)
aa=$(awk -F'\t' 'NR==1{for (i=1; i<=NF; i++) if ($i == "aaSubstitutions") print i}' metadata.tsv)

echo -e "date\tregion\thost\tNextstrain_clade\tpango_lineage\tdeletions\tinsertions\taaSubstitutions" > data.txt

awk -F'\t' -v date="$date" -v region="$region" -v host="$host" -v clade="$clade" -v lineage="$lineage" -v deletions="$deletions" -v insertions="$insertions" -v aa="$aa" 'NR>1{print $date"\t" $region"\t" $host"\t" $clade"\t" $lineage"\t" $deletions"\t" $insertions"\t" $aa}' metadata.tsv >> data.txt

rm metadata.tsv
