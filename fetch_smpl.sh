#!/bin/bash
# Installation script inspired by the excellent DECA code repository : https://github.com/yfeng95/DECA

urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

#Fetch SMPL data
echo -e "\nBefore you continue, you must register at https://smpl.is.tue.mpg.de/ and agree to the SMPL license terms."
read -p "Username (SMPL):" username
read -p "Password (SMPL):" password
username=$(urle $username)
password=$(urle $password)
mkdir -p ./data
echo -e "\nDownloading SMPL..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip&resume=1' -O './data/SMPL.zip' --no-check-certificate --continue 

#check that the file is more than 100MB
if [ ! -f ./data/SMPLH.tar.xz ] || [ $(stat -c%s ./data/SMPLH.tar.xz) -lt 100000000 ]; then
    echo "Error: SMPLH download failed or file is too small. Please check your credentials and try again."
    exit 1
fi

unzip ./data/SMPL.zip -d ./data/SMPL

#Rename stuffs
mkdir -p ./data/smpl
mv ./data/SMPL/SMPL_python_v.1.1.0/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl ./data/smpl/SMPL_FEMALE.pkl
mv ./data/SMPL/SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl ./data/smpl/SMPL_MALE.pkl
mv ./data/SMPL/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl ./data/smpl/SMPL_NEUTRAL.pkl
rm -rf ./data/SMPL
rm ./data/SMPL.zip