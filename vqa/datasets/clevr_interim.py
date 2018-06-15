import json
import os
import argparse
from collections import Counter

def get_subtype(split='train'):
    if split in ['train', 'val']:
        return split
    else:
        return 'test'

def interim(questions, split='train'):
    print('Interim', split)
    data = []
    for i in range(len(questions)):
        row = {}
        row['question_id'] = questions[i]['question_index']
        row['image_name'] = questions[i]['image_filename']
        row['question'] = questions[i]['question'].lower().strip()
        row['question_type'] = row['question'].split()[0]
        if split in ['train', 'val', 'trainval']:
            row['answer'] = questions[i]['answer'].lower().strip()
            row['question_family_id'] = questions[i]['question_family_index']
        data.append(row)
    return data

def selected_interim(questions, split='train'):
    print('Selected interim', split)
    path_question_type = './data/selected_question_types_clevr.txt'
    with open(path_question_type, 'r') as f:
        selected_question_types = f.read().splitlines()
        selected_question_types = [item for item in selected_question_types if not item.startswith('#')]
        if selected_question_types is None:
            raise Exception('Fail to load selected question types, please check the path')
    data = []
    ignore_counter = 0
    for i in range(len(questions)):
        row = {}
        row['question_id'] = questions[i]['question_index']
        row['image_name'] = questions[i]['image_filename']
        row['question'] = questions[i]['question'].lower().strip()
        row['question_type'] = row['question'].split()[0]
        if split in ['train', 'val', 'trainval']:
            row['answer'] = questions[i]['answer'].lower().strip()
            row['question_family_id'] = questions[i]['question_family_index']

        if row['question_type'] not in selected_question_types or row.get('answer', 'OMG').isdigit():
            ignore_counter += 1
            continue
        data.append(row)

    print('{} questions filtered.'.format(ignore_counter))
    return data

def clevr_interim(dir_clevr, select_questions=False):
    '''
    Put the VQA data into single json file in data/interim
    or train, val, trainval : [[question_id, image_name, question, MC_answer, answer] ... ]
    or test, test-dev :       [[question_id, image_name, question, MC_answer] ... ]
    '''

    interim_subfolder_name = 'selected_interim' if select_questions else 'interim'
    interim_function = selected_interim if select_questions else interim

    path_train_qa    = os.path.join(dir_clevr, interim_subfolder_name, 'train_questions_annotations.json')
    path_val_qa      = os.path.join(dir_clevr, interim_subfolder_name, 'val_questions_annotations.json')

    os.system('mkdir -p ' + os.path.join(dir_clevr, interim_subfolder_name))

    print('Loading annotations and questions...')
    questions_train   = json.load(open(os.path.join(dir_clevr, 'annotations', 'train.json'), 'r'))
    questions_val     = json.load(open(os.path.join(dir_clevr, 'annotations', 'val.json'), 'r'))

    data_train = interim_function(questions_train['questions'], 'train')
    print('Train size %d'%len(data_train))
    print('Write', path_train_qa)
    json.dump(data_train, open(path_train_qa, 'w'))

    data_val = interim_function(questions_val['questions'], 'val')
    print('Val size %d'%len(data_val))
    print('Write', path_val_qa)
    json.dump(data_val, open(path_val_qa, 'w'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_clevr', default='data/clevr', type=str, help='Path to clevr data directory')
    args = parser.parse_args()
    clevr_interim(args.dir_clevr)
