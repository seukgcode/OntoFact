from transformers import T5Tokenizer, T5ForConditionalGeneration,BartTokenizer, BartForConditionalGeneration, BartConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, OPTForCausalLM, GPT2Tokenizer,BloomForCausalLM,BloomTokenizerFast
# from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import argparse
import torch
import numpy as np
import time
import tqdm

class Bloom():
    def __init__(self, args):
        self.question_type = args.qt
        # self.model_path = args.T5
        
        self.tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-7b1", cache_dir = '/data01/c_x/bloomz-7b1',trust_remote_code=True)
        self.device = args.device
        self.model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-7b1", cache_dir = '/data01/c_x/bloomz-7b1',trust_remote_code=True).to(self.device)
    
    def build_start_question(self, ontoloty_triple, type):
        if type == 'triple':
            question = ['The general format of a knowledge triad is known to be (head entity, relation, tail entity). Determine whether the following knowledge triple is true: ({}, {}, {}).'.format(triple[0].replace('_',' '), triple[1].replace('_',' '), triple[2].replace('_',' ')) for triple in ontoloty_triple]
        elif type == 'template':
            question = ["The knowledge triple is generally expressed as: (subject, predicate, object). I would like you to transform the following knowledge triple into a question that can be answered with yes or no. \n({}, {}, {}). \nPlease tell me the question no other analysis.".format(triple[0].replace('_',' '), triple[1].replace('_',' '), triple[2].replace('_',' ')) for triple in ontoloty_triple]
        elif type == 'manual_template':
            question = ['Was the {} of {} {}? Perhaps you can give me the following answer: No, Yes. Do not give me reasons.'.format(triple[1].replace('_',' '), triple[0].replace('_',' '), triple[2].replace('_',' ')) for triple in ontoloty_triple]
        elif type == 'yes_no':
            template = """
                    Answer the following question, if you are uncertain about the answer, you should answer 'Unknown'.
                    Question: {}
                    Answer:"""
            question = [template.format(sentence) for sentence in ontoloty_triple]
        return question
    def build_answer_question(self, start_question):
        question = ['{} \n Please give me your answer: yes or no, do not give me other reasons.'.format(question[3]) for question in start_question]
        # question = ['Is the {} of {} {}? Perhaps you can give me the following answer: No, Yes, Uncertain. Do not give me reasons.'.format(triple[1].replace('_',' '), triple[0].replace('_',' '), triple[2].replace('_',' ')) for triple in ontoloty_triple]
        return question
        
    
    def generation(self,ontology_triple):
        question_tokenize = self.tokenizer.batch_encode_plus(self.build_start_question(ontology_triple,  'yes_no'), max_length = 200, pad_to_max_length=True, padding = "max_length", return_tensors = 'pt')
        generated_ids = self.model.generate(input_ids = question_tokenize['input_ids'].to(self.device), attention_mask = question_tokenize['attention_mask'].to(self.device), max_length=210)
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = [r.split('Answer:')[1].strip() for r in response]
        # print(response)

        # question_tokenize = self.tokenizer.batch_encode_plus(self.build_answer_question(response), max_length = 128, pad_to_max_length=True, padding = "max_length", return_tensors = 'pt')
        # generated_ids = self.model.generate(input_ids = question_tokenize['input_ids'].to(self.device), attention_mask = question_tokenize['attention_mask'].to(self.device), max_new_tokens=128)
        # response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        return list(zip(ontology_triple, response))

class T0():
    def __init__(self, args):
        self.question_type = args.qt
        # self.model_path = args.T5

        self.tokenizer = AutoTokenizer.from_pretrained('/data01/c_x/T0')
        self.device = args.device
        self.model = AutoModelForSeq2SeqLM.from_pretrained('/data01/c_x/T0').to(self.device)
    

    def build_answer_question(self, start_question):
        template = """
                    Answer the following question, if you are uncertain about the answer, you should answer 'Unknown'.
                    Question: {}"""
        question = [template.format(question) for question in start_question]
        # question = ['Is the {} of {} {}? Perhaps you can give me the following answer: No, Yes, Uncertain. Do not give me reasons.'.format(triple[1].replace('_',' '), triple[0].replace('_',' '), triple[2].replace('_',' ')) for triple in ontoloty_triple]
        return question
        
    
    def generation(self,ontology_triple):
       

        question_tokenize = self.tokenizer.batch_encode_plus(self.build_answer_question(ontology_triple), max_length = 128, pad_to_max_length=True, padding = "max_length", return_tensors = 'pt',truncation=True)
        generated_ids = self.model.generate(input_ids = question_tokenize['input_ids'].to(self.device), attention_mask = question_tokenize['attention_mask'].to(self.device), max_new_tokens=150)
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        response = [r.strip() for r in response]
        
        return list(zip(ontology_triple, response))        

class FLAN_T5_XXL():
    def __init__(self, args):
        self.question_type = args.qt
        # self.model_path = args.T5

        self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xxl', cache_dir = '/data01/c_x/flan-t5-xxl')
        self.device = args.device
        self.model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-xxl', cache_dir = '/data01/c_x/flan-t5-xxl').to(self.device)
    
    def build_answer_question(self, start_question):
        template = """
                    Answer the following question, if you are uncertain about the answer, you should answer 'Unknown'.
                    Question: {}"""
        question = [template.format(question[3]) for question in start_question]
        # question = ['Is the {} of {} {}? Perhaps you can give me the following answer: No, Yes, Uncertain. Do not give me reasons.'.format(triple[1].replace('_',' '), triple[0].replace('_',' '), triple[2].replace('_',' ')) for triple in ontoloty_triple]
        return question
        
    
    def generation(self,ontology_triple):
        question_tokenize = self.tokenizer.batch_encode_plus(self.build_answer_question(ontology_triple), max_length = 128, pad_to_max_length=True, padding = "max_length", return_tensors = 'pt',truncation=True)
        generated_ids = self.model.generate(input_ids = question_tokenize['input_ids'].to(self.device), attention_mask = question_tokenize['attention_mask'].to(self.device), max_new_tokens=150)
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        response = [r.strip() for r in response]
        
        return list(zip(ontology_triple, response))

class alpaca_7b():
    def __init__(self, args):
        self.question_type = args.qt
        self.tokenizer = AutoTokenizer.from_pretrained('/data01/c_x/alpaca-7b-wdiff/new', padding_side='left')
        self.device = args.device
        self.model = AutoModelForCausalLM.from_pretrained('/data01/c_x/alpaca-7b-wdiff/new').to(self.device)

    def build_start_question(self, ontoloty_triple, type):
    
        question = [(
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nAnswer the given question.\n\n### Input:\n{}\n\n### Response:".format(sentence[3])
        ) for sentence in ontoloty_triple]
        # question = ['{} Please give me the following answer: Yes or No. Answer:'.format(
        #     sentence) for sentence in ontoloty_triple]
        return question

    def generation(self, ontology_triple):

        question_tokenize = self.tokenizer.batch_encode_plus(self.build_start_question(
            ontology_triple,  'yes_no'), max_length=180, pad_to_max_length=True, padding="max_length", return_tensors='pt',truncation=True)
        generated_ids = self.model.generate(input_ids=question_tokenize['input_ids'].to(
            self.device), attention_mask=question_tokenize['attention_mask'].to(self.device), max_new_tokens=10)
        response = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)
        response = [r.split('Response:')[-1].strip() for r in response]

        return list(zip(ontology_triple, response))

class OPT_13b():
    def __init__(self, args):
        self.question_type = args.qt
        self.device = args.device
        if args.size == 13:
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                "facebook/opt-13b", cache_dir='/data01/c_x/OPT-13B', padding_side='left')
            
            self.model = OPTForCausalLM.from_pretrained(
                "facebook/opt-13b", cache_dir='/data01/c_x/OPT-13B').to(self.device)
        if args.size == 6:
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                "/data01/c_x/opt-6.7B", padding_side='left')
            
            self.model = OPTForCausalLM.from_pretrained(
                "/data01/c_x/opt-6.7B").to(self.device)

    def build_start_question(self, ontoloty_triple, type):
        if type == 'triple':
            question = ['The general format of a knowledge triad is known to be (head entity, relation, tail entity). Determine whether the following knowledge triple is true: ({}, {}, {}).'.format(
                triple[0].replace('_', ' '), triple[1].replace('_', ' '), triple[2].replace('_', ' ')) for triple in ontoloty_triple]
        elif type == 'template':
            question = ["The knowledge triple is generally expressed as: (subject, predicate, object). I would like you to transform the following knowledge triple into a question that can be answered with yes or no. \n({}, {}, {}). \nPlease tell me the question no other analysis.".format(
                triple[0].replace('_', ' '), triple[1].replace('_', ' '), triple[2].replace('_', ' ')) for triple in ontoloty_triple]
        elif type == 'manual_template':
            question = ['Was the {} of {} {}? Perhaps you can give me the following answer: No, Yes. Do not give me reasons.'.format(
                triple[1].replace('_', ' '), triple[0].replace('_', ' '), triple[2].replace('_', ' ')) for triple in ontoloty_triple]
        elif type == 'yes_no':
            question = [(
                "Question:{} Answer:".format(sentence[3])
            ) for sentence in ontoloty_triple]
            # question = ['{} Please give me the following answer: Yes or No. Answer:'.format(
            #     sentence) for sentence in ontoloty_triple]
        return question

    def generation(self, ontology_triple):

        question_tokenize = self.tokenizer.batch_encode_plus(self.build_start_question(
            ontology_triple,  'yes_no'), max_length=128, pad_to_max_length=True, padding="max_length", return_tensors='pt', truncation=True)
        generated_ids = self.model.generate(input_ids=question_tokenize['input_ids'].to(
            self.device), attention_mask=question_tokenize['attention_mask'].to(self.device), max_length=150)
        response = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False, max_new_tokens = 150)
        response = [r.split('Answer:')[1].strip().replace('\n',' ') for r in response]

        return list(zip(ontology_triple, response))

class GPT_6j():
    def __init__(self, args):
        self.device = 'cuda:0'
        self.question_type = args.qt
        self.tokenizer = AutoTokenizer.from_pretrained(
            '/data01/c_x/GPT/GPT-J-6B',padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = args.device
        self.model = AutoModelForCausalLM.from_pretrained(
            '/data01/c_x/GPT/GPT-J-6B').to(self.device)
        

    def build_start_question(self, ontoloty_triple, type):
        template = """
                Question: {}
                Answer:"""
        question = [template.format(sentence[3]) for sentence in ontoloty_triple]
        return question
    def generation(self,ontology_triple):
        question_tokenize = self.tokenizer.batch_encode_plus(self.build_start_question(ontology_triple,  'yes_no'), max_length = 200, pad_to_max_length=True, padding = "max_length", return_tensors = 'pt',truncation=True)
        generated_ids = self.model.generate(input_ids = question_tokenize['input_ids'].to(self.device), attention_mask = question_tokenize['attention_mask'].to(self.device), max_length=210,pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = [r.split('Answer:')[1].strip() for r in response]
        return list(zip(ontology_triple, response))



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",  type=str, default='all_entity_0_demo')
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)
    parser.add_argument("-c", "--cluster", help="the type of cluster algorithm", type=str, default='K-means')
    parser.add_argument("-u", "--unified", help="the type of cluster algorithm", action='store_false')
    parser.add_argument("--device", type=str, default='0')
    parser.add_argument("--LLMType", type=str, default='T0')
    parser.add_argument("--qt", help='Question Type such as: triple、template、manual_template', type=str, default='triple')
    parser.add_argument("--data_path", help='Question Type such as: triple、template、manual_template', type=str, default='triple')
    parser.add_argument("--save_path", help='Question Type such as: triple、template、manual_template', type=str, default='triple')
    parser.add_argument("--batch_size", help='Question Type such as: triple、template、manual_template', type=int, default=8)
    parser.add_argument("--size", help='Question Type such as: triple、template、manual_template', type=int, default=13)

    args = parser.parse_args()
    args.device = 'cuda:{}'.format(args.device)
    # GPT_6j(args)
    if args.LLMType == 'T0':
        model = T0(args)
    if args.LLMType == 'Alphaca':
        model = alpaca_7b(args)
    if args.LLMType == 'FLAN_T5_XXL':
        model = FLAN_T5_XXL(args)
    if args.LLMType == 'OPT':
        model = OPT_13b(args)
    if args.LLMType == 'GPTJ':
        model = GPT_6j(args)
    all_data = []
    # /data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/数据量缩小/final_triple_question.txt
    with open(args.data_path) as f:
        for i in f.readlines():
            h,r,t,q = i.strip().split('\t')
            all_data.append([h,r,t,q])
    all_data = all_data
    LLM_result = open(args.save_path, 'w+')
    bath_size = args.batch_size
    n_batch = len(all_data) // bath_size + (len(all_data) % bath_size > 0)
    # print(n_batch)
    for i in tqdm.tqdm(range(n_batch)):
        start_time = time.time()
        start = i*bath_size
        end = min(len(all_data), (i+1)*bath_size)
        batch_data = np.array(all_data)[np.arange(start, end)]
        s = time.time()
        result = model.generation(batch_data)
        print(time.time()-s)
        LLM_result.writelines(['\t'.join(j[0]) + '\t' + j[1] + '\n' for j in result])
    LLM_result.close()