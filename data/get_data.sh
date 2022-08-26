#!/bin/sh

## parameters
FTP_PATH="https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/taxonomic_divisions/"
FILE_TAXOM="uniprot_trembl_human.dat.gz"

# download
wget ${FTP_PATH}${FILE_TAXOM}