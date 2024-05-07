python /data01/c_x/ChatGPT_Demo/LLM_sh.py --batch_size 8 --LLMType  T0  --data_path /data01/c_x/all_result/bios_eng/data/20230714140951_english_only_triple_one_question.txt    --save_path /data01/c_x/all_result/bios_eng/result/T0.txt

python /data01/c_x/ChatGPT_Demo/LLM_sh.py --batch_size 8 --LLMType  T0  --data_path /data01/c_x/all_result/bios_eng/data/20230714140951_english_only_triple_one_question_NA.txt    --save_path /data01/c_x/all_result/bios_eng/result/T0-NA.txt

python /data01/c_x/ChatGPT_Demo/LLM_sh.py --batch_size 32 --LLMType  Alphaca  --data_path /data01/c_x/all_result/bios_eng/data/20230714140951_english_only_triple_one_question.txt    --save_path /data01/c_x/all_result/bios_eng/result/Alphaca-7b.txt

python /data01/c_x/ChatGPT_Demo/LLM_sh.py --batch_size 32 --LLMType  Alphaca  --data_path /data01/c_x/all_result/bios_eng/data/20230714140951_english_only_triple_one_question_NA.txt    --save_path /data01/c_x/all_result/bios_eng/result/Alphaca-7b-NA.txt

python /data01/c_x/ChatGPT_Demo/LLM_sh.py --batch_size 300 --LLMType  FLAN_T5_XXL  --data_path /data01/c_x/all_result/bios_eng/data/20230714140951_english_only_triple_one_question.txt    --save_path /data01/c_x/all_result/bios_eng/result/FLAN_T5_XXL.txt

python /data01/c_x/ChatGPT_Demo/LLM_sh.py --batch_size 300 --LLMType  FLAN_T5_XXL  --data_path /data01/c_x/all_result/bios_eng/data/20230714140951_english_only_triple_one_question_NA.txt    --save_path /data01/c_x/all_result/bios_eng/result/FLAN_T5_XXL-NA.txt

python /data01/c_x/ChatGPT_Demo/LLM_sh.py --batch_size 32 --LLMType  OPT  --data_path /data01/c_x/all_result/bios_eng/data/20230714140951_english_only_triple_one_question.txt    --save_path /data01/c_x/all_result/bios_eng/result/OPT13B.txt --size 13

python /data01/c_x/ChatGPT_Demo/LLM_sh.py --batch_size 32 --LLMType  OPT  --data_path /data01/c_x/all_result/bios_eng/data/20230714140951_english_only_triple_one_question_NA.txt    --save_path /data01/c_x/all_result/bios_eng/result/OPT13B-NA.txt --size 13

python /data01/c_x/FastChat/fastchat/llm_judge/gen_model_answer.py  --model-path /data01/c_x/vicuna-13b-v1.3 --model-id vicuna-13b-v1.3 --data_onto /data01/c_x/all_result/bios_eng/data/20230714140951_english_only_triple_one_question.txt --save_path /data01/c_x/all_result/bios_eng/result/vicuna.txt

python /data01/c_x/FastChat/fastchat/llm_judge/gen_model_answer.py  --model-path /data01/c_x/vicuna-13b-v1.3 --model-id vicuna-13b-v1.3 --data_onto /data01/c_x/all_result/bios_eng/data/20230714140951_english_only_triple_one_question_NA.txt --save_path /data01/c_x/all_result/bios_eng/result/vicuna-NA.txt

python /data01/c_x/ChatGPT_Demo/LLM_sh.py --batch_size 64 --LLMType  OPT  --data_path /data01/c_x/all_result/bios_eng/data/20230714140951_english_only_triple_one_question.txt    --save_path /data01/c_x/all_result/bios_eng/result/OPT6B.txt --size 6

python /data01/c_x/ChatGPT_Demo/LLM_sh.py --batch_size 64 --LLMType  OPT  --data_path /data01/c_x/all_result/bios_eng/data/20230714140951_english_only_triple_one_question_NA.txt    --save_path /data01/c_x/all_result/bios_eng/result/OPT6B-NA.txt --size 6

python /data01/c_x/ChatGPT_Demo/LLM_sh.py --batch_size 50   --LLMType  Alphaca  --data_path /data01/c_x/all_result/bios_eng/data/20230714140951_english_only_triple_one_question.txt    --save_path /data01/c_x/all_result/bios_eng/result/Alphaca-7b_test_time.txt

python /data01/c_x/ChatGPT_Demo/LLM_sh.py --batch_size 30 --LLMType  T0  --data_path /data01/c_x/all_result/bios_eng/data/20230714140951_english_only_triple_one_question.txt    --save_path /data01/c_x/all_result/bios_eng/result/T0_test.txt

python /data01/c_x/ChatGPT_Demo/LLM_sh.py --batch_size 50 --LLMType  GPTJ  --data_path /data01/c_x/all_result/bios_eng/data/20230714140951_english_only_triple_one_question.txt    --save_path /data01/c_x/all_result/bios_eng/result/GPTJ_test_time.txt

python /data01/c_x/ChatGPT_Demo/LLM_sh.py --batch_size 32 --LLMType  OPT  --data_path /data01/c_x/all_result/yago/original_file/test-NA.txt    --save_path /data01/c_x/all_result/yago/OPT13B-NA-fulu.txt --size 13