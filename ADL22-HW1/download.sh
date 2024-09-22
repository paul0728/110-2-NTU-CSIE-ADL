#create prep folder
if [ ! -d prep ] ; then
  mkdir prep
  mkdir -p prep/intent
  mkdir -p prep/slot
fi


#create ckpt folder
if [ ! -d ckpt ] ; then
  mkdir ckpt
  mkdir -p ckpt/intent
  mkdir -p ckpt/slot
fi




#download preprocessed file

##intent classification

wget https://www.dropbox.com/sh/cyxjg6pindm2c99/AAAveWrTX4LTrd-BmB0XOIj2a?dl=1 -O prep/intent/intent_prep.zip
unzip prep/intent/intent_prep.zip -d prep/intent
rm  -f prep/intent/intent_prep.zip




##slot tagging

wget https://www.dropbox.com/sh/tcvfpyn90r1rdtf/AADtZEPILu79YiwDmulkl_iga?dl=1 -O prep/slot/slot_prep.zip
unzip prep/slot/slot_prep.zip -d prep/slot
rm  -f prep/slot/slot_prep.zip

#download best ckpt

##intent classification

if [ ! -d ckpt/intent/2BILSTM_nooov_mw_10000_ml_28 ] ; then
  wget https://www.dropbox.com/sh/t5zxyqpn2w3ve6z/AAA66e6ZL6HnWDhvavYwo4sVa?dl=1 -O ckpt/intent/2BILSTM_nooov_mw_10000_ml_28.zip
  unzip ckpt/intent/2BILSTM_nooov_mw_10000_ml_28.zip -d ckpt/intent/2BILSTM_nooov_mw_10000_ml_28
fi
rm  -f ckpt/intent/2BILSTM_nooov_mw_10000_ml_28.zip



##slot tagging

if [ ! -d ckpt/slot/2BILSTM_mw_1002_ml_35 ] ; then
  wget https://www.dropbox.com/sh/tsgryxw1swsaq4y/AAA-HhnBrLA1z6RvQwlTJMGWa?dl=1 -O ckpt/slot/2BILSTM_mw_1002_ml_35.zip
  unzip ckpt/slot/2BILSTM_mw_1002_ml_35.zip -d ckpt/slot/2BILSTM_mw_1002_ml_35
fi
rm  -f ckpt/slot/2BILSTM_mw_1002_ml_35.zip





