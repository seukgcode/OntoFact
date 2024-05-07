from transformers import T5Tokenizer, T5ForConditionalGeneration,BartTokenizer, BartForConditionalGeneration, BartConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from transformers.generation.utils import GenerationConfig
# from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import argparse
import torch
import numpy as np
import time
import tqdm


class Chinese_Alpaca_13B():
    def __init__(self):
        # self.question_type = args.qt
        self.tokenizer = AutoTokenizer.from_pretrained("/data01/c_x/cn-model/Chatglm-v2/chatglm2-6b", trust_remote_code=True)
        # self.device = args.device
        self.model = AutoModelForCausalLM.from_pretrained("/data01/c_x/cn-model/Chatglm-v2/chatglm2-6b", trust_remote_code=True).to('cuda:0')
    def prompt(self,text):
        return "[Round {}]\n\n任务：在不改变句意的情况下，润色下列句子。\n\n 句子：{}的{}是{}吗？\n\n不需要给出分析过程，润色后的句子：".format(1, text[0], text[1], text[2])
    
    def generation(self, triple):
        queries = [self.prompt(query) for query in triple]
        inputs = self.tokenizer(queries, padding=True, return_tensors="pt", truncation=True, max_length=128).to('cuda')
        outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=150)
        intermediate_outputs = []
        for idx in range(len(outputs)):
            output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
            response = self.tokenizer.decode(output)
            intermediate_outputs.append(response)
        answer_texts = [text[0] + '\t' + text[1] + '\t' + text[2] + '\t' + intermediate + "\n" for text, intermediate in zip(triple, intermediate_outputs)]
        print(answer_texts)
        # print(response)

class ChatGLMV2():
    def __init__(self):
        # self.question_type = args.qt
        self.tokenizer = AutoTokenizer.from_pretrained("/data01/c_x/cn-model/Chatglm-v2/chatglm2-6b", trust_remote_code=True)
        # self.device = args.device
        self.model = AutoModelForCausalLM.from_pretrained("/data01/c_x/cn-model/Chatglm-v2/chatglm2-6b", trust_remote_code=True).to('cuda:0')
    def prompt(self,text):
        return "[Round {}]\n\n任务：在不改变句意的情况下，润色下列句子。\n\n 句子：{}的{}是{}吗？\n\n不需要给出分析过程，润色后的句子：".format(1, text[0], text[1], text[2])
    
    def generation(self, triple):
        queries = [self.prompt(query) for query in triple]
        inputs = self.tokenizer(queries, padding=True, return_tensors="pt", truncation=True, max_length=128).to('cuda')
        outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=150)
        intermediate_outputs = []
        for idx in range(len(outputs)):
            output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
            response = self.tokenizer.decode(output)
            intermediate_outputs.append(response)
        answer_texts = [text[0] + '\t' + text[1] + '\t' + text[2] + '\t' + intermediate + "\n" for text, intermediate in zip(triple, intermediate_outputs)]
        print(answer_texts)
        # print(response)

class moss():
    def __init__(self):
        # self.question_type = args.qt
        # self.tokenizer = AutoTokenizer.from_pretrained("/data01/c_x/cn-model/Chatglm-v2/chatglm2-6b", trust_remote_code=True)
        # self.device = args.device
        self.tokenizer = AutoTokenizer.from_pretrained("fnlp/moss-moon-003-sft-plugin", trust_remote_code=True, cache_dir = '/data01/c_x/cn-model/moss')
        self.model = AutoModelForCausalLM.from_pretrained("fnlp/moss-moon-003-sft-plugin", trust_remote_code=True, cache_dir = '/data01/c_x/cn-model/moss').to('cuda:0')
        exit()
    def prompt(self,text):
        return "[Round {}]\n\n任务：在不改变句意的情况下，润色下列句子。\n\n 句子：{}的{}是{}吗？\n\n不需要给出分析过程，润色后的句子：".format(1, text[0], text[1], text[2])
    
    def generation(self, triple):
        queries = [self.prompt(query) for query in triple]
        inputs = self.tokenizer(queries, padding=True, return_tensors="pt", truncation=True, max_length=128).to('cuda')
        outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=150)
        intermediate_outputs = []
        for idx in range(len(outputs)):
            output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
            response = self.tokenizer.decode(output)
            intermediate_outputs.append(response)
        answer_texts = [text[0] + '\t' + text[1] + '\t' + text[2] + '\t' + intermediate + "\n" for text, intermediate in zip(triple, intermediate_outputs)]
        print(answer_texts)
        # print(response)

class bell13B():
    def __init__(self):
        # self.question_type = args.qt
        self.tokenizer = LlamaTokenizer.from_pretrained("/data01/c_x/cn-model/bell/to_finetuned_model13B")
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 2
        self.tokenizer.bos_token_id = 1
        self.tokenizer.padding_side = "left"
        # self.device = args.device
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = LlamaForCausalLM.from_pretrained("/data01/c_x/cn-model/bell/to_finetuned_model13B").to('cuda:0')
    def prompt(self,text):
        return "Human：简要直接给出下列问题的答案。\n\n 问题：{}\n\nBelle：".format(text)
    
    def generation(self, triple):
        queries = [self.prompt(query[3]) for query in triple]
        inputs = self.tokenizer(queries, padding=True, return_tensors="pt", truncation=True, max_length=128).to('cuda')
        outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=20, eos_token_id=2, bos_token_id=1, pad_token_id=0)
        intermediate_outputs = []
        for idx in range(len(outputs)):
            output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
            response = self.tokenizer.decode(output,skip_special_tokens=True, clean_up_tokenization_spaces=False)
            intermediate_outputs.append(response)
        answer_texts = ['\t'.join(text) + '\t' + intermediate.replace('\n',' ') + "\n" for text, intermediate in zip(triple, intermediate_outputs)]
        # print(answer_texts)
        return answer_texts
        # print(response)
class bell7B():
    def __init__(self):
        # self.question_type = args.qt
        self.tokenizer = LlamaTokenizer.from_pretrained("/data01/c_x/cn-model/bell/to_finetuned_model7B")
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 2
        self.tokenizer.bos_token_id = 1
        self.tokenizer.padding_side = "left"
        # self.device = args.device
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = LlamaForCausalLM.from_pretrained("/data01/c_x/cn-model/bell/to_finetuned_model7B").to('cuda:0')
    def prompt(self,text):
        return "Human：简单回答下列问题，请直接给出答案。\n\n问题：{}\n\nBelle：".format(text)
    
    def generation(self, triple):
        queries = [self.prompt(query[3]) for query in triple]
        inputs = self.tokenizer(queries, padding=True, return_tensors="pt", truncation=True, max_length=128).to('cuda')
        outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=20, eos_token_id=2, bos_token_id=1, pad_token_id=0)
        intermediate_outputs = []
        for idx in range(len(outputs)):
            output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
            response = self.tokenizer.decode(output,skip_special_tokens=True, clean_up_tokenization_spaces=False)
            intermediate_outputs.append(response)
        answer_texts = ['\t'.join(text) + '\t' + intermediate.replace('\n',' ') + "\n" for text, intermediate in zip(triple, intermediate_outputs)]
        # print(answer_texts)
        return answer_texts
        # print(response)

class panda():
    def __init__(self):
        # self.question_type = args.qt
        self.tokenizer = AutoTokenizer.from_pretrained("chitanda/llama-panda-zh-coig-7b-delta",cache_dir = '/data01/c_x/cn-model/panda/7b')
        self.model = AutoModelForCausalLM.from_pretrained("chitanda/llama-panda-zh-coig-7b-delta",cache_dir = '/data01/c_x/cn-model/panda/7b')
        exit()
    def prompt(self,text):
        return "Human：简单回答下列问题，请直接给出答案。\n\n问题：{}\n\nBelle：".format(text)
    
    def generation(self, triple):
        queries = [self.prompt(query[3]) for query in triple]
        inputs = self.tokenizer(queries, padding=True, return_tensors="pt", truncation=True, max_length=128).to('cuda')
        outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=20, eos_token_id=2, bos_token_id=1, pad_token_id=0)
        intermediate_outputs = []
        for idx in range(len(outputs)):
            output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
            response = self.tokenizer.decode(output,skip_special_tokens=True, clean_up_tokenization_spaces=False)
            intermediate_outputs.append(response)
        answer_texts = ['\t'.join(text) + '\t' + intermediate.replace('\n',' ') + "\n" for text, intermediate in zip(triple, intermediate_outputs)]
        # print(answer_texts)
        return answer_texts
        # print(response)

class baichuan7B():
    def __init__(self):
        # self.question_type = args.qt
        # self.tokenizer = AutoTokenizer.from_pretrained("/data01/c_x/cn-model/baichuan/7B",trust_remote_code=True)
        # # self.device = args.device
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.model = AutoModelForCausalLM.from_pretrained("/data01/c_x/cn-model/baichuan/7B",trust_remote_code=True).to('cuda:0')

        self.tokenizer = AutoTokenizer.from_pretrained("/data01/c_x/cn-model/baichuan/Baichuan-13B-Chat",trust_remote_code=True,padding_side='left')
        # self.device = args.device
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained("/data01/c_x/cn-model/baichuan/Baichuan-13B-Chat",trust_remote_code=True).to('cuda:0')
        self.model.generation_config = GenerationConfig.from_pretrained("/data01/c_x/cn-model/baichuan/Baichuan-13B-Chat")
    def prompt(self,text):
        return "直接给出下列问题答案。\n\n问题：{}\n\n答案：".format(text)
    
    def generation(self, triple):
        queries = [self.prompt(query[3]) for query in triple]
        inputs = self.tokenizer(queries, padding=True, return_tensors="pt", truncation=True, max_length=128).to('cuda')
        outputs = self.model.generate(**inputs, max_new_tokens=20)
        intermediate_outputs = []
        for idx in range(len(outputs)):
            output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
            response = self.tokenizer.decode(output,skip_special_tokens=True, clean_up_tokenization_spaces=False)
            intermediate_outputs.append(response.replace('\n',' '))
        answer_texts = ['\t'.join(text) + '\t' + intermediate.replace('\n',' ') + "\n" for text, intermediate in zip(triple, intermediate_outputs)]
        return answer_texts
        # print(response)

        # question_tokenize = self.tokenizer.batch_encode_plus(self.build_start_question(ontology_triple,  'yes_no'), max_length = 200, pad_to_max_length=True, padding = "max_length", return_tensors = 'pt',truncation=True)
        # generated_ids = self.model.generate(input_ids = question_tokenize['input_ids'].to(self.device), attention_mask = question_tokenize['attention_mask'].to(self.device), max_length=210,pad_token_id=self.tokenizer.eos_token_id)
        # response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # response = [r.split('答案：')[1].strip() for r in response]
        # return list(zip(triple, response))

if __name__ == '__main__':
    # a = bell13B()
    # a.generation([('白小华（准煤公司生产调度员）','民族','回族'),('AOpen AK77 Plus','CPU插槽','Socket A')])
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--dataset",  type=str, default='all_entity_0_demo')
    # parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)
    # parser.add_argument("-c", "--cluster", help="the type of cluster algorithm", type=str, default='K-means')
    # parser.add_argument("-u", "--unified", help="the type of cluster algorithm", action='store_false')
    # parser.add_argument("--device", type=str, default='0')
    # parser.add_argument("--LLMType", type=str, default='T0')
    # parser.add_argument("--qt", help='Question Type such as: triple、template、manual_template', type=str, default='triple')

    # args = parser.parse_args()
    # args.device = 'cuda:{}'.format(args.device)
    # # GPT_6j(args)
    model = baichuan7B()
    all_data = []
    #/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/final_data/202307311353_only_triple_NA.txt
    with open('/data01/c_x/all_result/cn_dbpedia/result/ChatGPT_result.txt') as f:
        for i in f.readlines():
            h,r,t,q,_ = i.strip().split('\t')
            all_data.append([h,r,t,q])
    all_data = all_data
    LLM_result = open('/data01/c_x/all_result/cn_dbpedia/result/baichuan13B_NA.txt', 'a')
    bath_size = 32
    n_batch = len(all_data) // bath_size + (len(all_data) % bath_size > 0)
    # print(n_batch)
    all_result = []
    for i in tqdm.tqdm(range(n_batch)):
        start_time = time.time()
        start = i*bath_size
        end = min(len(all_data), (i+1)*bath_size)
        batch_data = np.array(all_data)[np.arange(start, end)]
        result = model.generation(batch_data)
        # all_result.append(result)
        # LLM_result.writelines(['\t'.join(j[0]) + '\t' + j[1] + '\n' for j in result])
        LLM_result.writelines(result)
    # for i in all_result:
    #     LLM_result.writelines(i)
    LLM_result.close()