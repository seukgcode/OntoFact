python ./run/training_model2_scai2.py --kg1f ./data/yago_4.5/triple.txt --kg2f ./data/yago_4.5/db_onto_train.txt --modelname ./yago_4.5_test  --method hole --bridge CMP-linear --alignf ./data/yago_4.5/db_InsType_train.txt

python ./run/training_model2_scai2.py --kg1f ./data/cn-dbpedia/triple.txt --kg2f ./data/cn-dbpedia/db_onto_train.txt --modelname ./cn-dbpedia-202308131206  --method hole --bridge CMP-linear --alignf ./data/cn-dbpedia/db_InsType_train.txt

python ./run/training_model2_scai2.py --kg1f ./data/dbpedia_final/triple.txt --kg2f ./data/dbpedia_final/db_onto_train.txt --modelname ./dbpedia-202308061707  --method hole --bridge CMP-linear --alignf ./data/dbpedia_final/db_InsType_train.txt --GPU 1

python ./run/training_model2_scai2.py --kg1f ./data/bios_eng/triple.txt --kg2f ./data/bios_eng/db_onto_train.txt --modelname ./bioseng_202308061500  --method hole --bridge CMP-linear --alignf ./data/bios_eng/db_InsType_train.txt

python ./run/training_model2_scai2.py --kg1f ./data/bios_chs/triple.txt --kg2f ./data/bios_chs/db_onto_train.txt --modelname ./bioschs_202308152116  --method hole --bridge CMP-linear --alignf ./data/bios_chs/db_InsType_train.txt