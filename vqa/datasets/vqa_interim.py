import json
import os
import argparse
from collections import Counter

def get_subtype(split='train'):
    if split in ['train', 'val']:
        return split + '2014'
    else:
        return 'test2015'

def get_image_name_old(subtype='train2014', image_id='1', format='%s/COCO_%s_%012d.jpg'):
    return format%(subtype, subtype, image_id)

def get_image_name(subtype='train2014', image_id='1', format='COCO_%s_%012d.jpg'):
    return format%(subtype, image_id)

def interim(questions, split='train', annotations=[]):
    print('Interim', split)
    data = []
    for i in range(len(questions)):
        row = {}
        row['question_id'] = questions[i]['question_id']
        row['image_name'] = get_image_name(get_subtype(split), questions[i]['image_id'])
        row['question'] = questions[i]['question']
        row['MC_answer'] = questions[i]['multiple_choices']
        if split in ['train', 'val', 'trainval']:
            row['answer'] = annotations[i]['multiple_choice_answer']
            answers = []
            for ans in annotations[i]['answers']:
                answers.append(ans['answer'])
            row['answers_occurence'] = Counter(answers).most_common()
        data.append(row)
    return data

def selected_interim(questions, split='train', annotations=[]):
    print('Selected interim', split)
    path_question_type = './data/selected_question_types.txt'
    with open(path_question_type, 'r') as f:
        selected_question_types = f.read().splitlines()
        selected_question_types = [item for item in selected_question_types if not item.startswith('#')]
        if selected_question_types is None:
            raise Exception('Fail to load selected question types, please check the path')
    data = []
    ignore_counter = 0
    for i in range(len(questions)):
        if split in ['train', 'val', 'trainval'] and annotations[i]['question_type'] not in selected_question_types:
            ignore_counter += 1
            continue
        row = {}
        row['question_id'] = questions[i]['question_id']
        row['image_name'] = get_image_name(get_subtype(split), questions[i]['image_id'])
        row['question'] = questions[i]['question']
        row['MC_answer'] = questions[i]['multiple_choices']
        if split in ['train', 'val', 'trainval']:
            row['answer'] = annotations[i]['multiple_choice_answer']
            answers = []
            for ans in annotations[i]['answers']:
                answers.append(ans['answer'])
            row['answers_occurence'] = Counter(answers).most_common()
        data.append(row)

    print('{} questions filtered.'.format(ignore_counter))
    return data

def vqa_interim(dir_vqa, select_questions=False):
    '''
    Put the VQA data into single json file in data/interim
    or train, val, trainval : [[question_id, image_name, question, MC_answer, answer] ... ]
    or test, test-dev :       [[question_id, image_name, question, MC_answer] ... ]
    '''

    interim_subfolder_name = 'selected_interim' if select_questions else 'interim'
    interim_function = selected_interim if select_questions else interim

    path_train_qa    = os.path.join(dir_vqa, interim_subfolder_name, 'train_questions_annotations.json')
    path_val_qa      = os.path.join(dir_vqa, interim_subfolder_name, 'val_questions_annotations.json')
    path_trainval_qa = os.path.join(dir_vqa, interim_subfolder_name, 'trainval_questions_annotations.json')
    path_test_q      = os.path.join(dir_vqa, interim_subfolder_name, 'test_questions.json')
    path_testdev_q   = os.path.join(dir_vqa, interim_subfolder_name, 'testdev_questions.json')

    os.system('mkdir -p ' + os.path.join(dir_vqa, interim_subfolder_name))

    print('Loading annotations and questions...')
    annotations_train = json.load(open(os.path.join(dir_vqa, 'raw', 'annotations', 'mscoco_train2014_annotations.json'), 'r'))
    annotations_val   = json.load(open(os.path.join(dir_vqa, 'raw', 'annotations', 'mscoco_val2014_annotations.json'), 'r'))
    questions_train   = json.load(open(os.path.join(dir_vqa, 'raw', 'annotations', 'MultipleChoice_mscoco_train2014_questions.json'), 'r'))
    questions_val     = json.load(open(os.path.join(dir_vqa, 'raw', 'annotations', 'MultipleChoice_mscoco_val2014_questions.json'), 'r'))
    questions_test    = json.load(open(os.path.join(dir_vqa, 'raw', 'annotations', 'MultipleChoice_mscoco_test2015_questions.json'), 'r'))
    questions_testdev = json.load(open(os.path.join(dir_vqa, 'raw', 'annotations', 'MultipleChoice_mscoco_test-dev2015_questions.json'), 'r'))

    data_train = interim_function(questions_train['questions'], 'train', annotations_train['annotations'])
    print('Train size %d'%len(data_train))
    print('Write', path_train_qa)
    json.dump(data_train, open(path_train_qa, 'w'))

    data_val = interim_function(questions_val['questions'], 'val', annotations_val['annotations'])
    print('Val size %d'%len(data_val))
    print('Write', path_val_qa)
    json.dump(data_val, open(path_val_qa, 'w'))

    print('Concat. train and val')
    data_trainval = data_train + data_val
    print('Trainval size %d'%len(data_trainval))
    print('Write', path_trainval_qa)
    json.dump(data_trainval, open(path_trainval_qa, 'w'))

    data_testdev = interim_function(questions_testdev['questions'], 'testdev')
    print('Testdev size %d'%len(data_testdev))
    print('Write', path_testdev_q)
    json.dump(data_testdev, open(path_testdev_q, 'w'))

    data_test = interim_function(questions_test['questions'], 'test')
    print('Test size %d'%len(data_test))
    print('Write', path_test_q)
    json.dump(data_test, open(path_test_q, 'w'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_vqa', default='data/vqa', type=str, help='Path to vqa data directory')
    args = parser.parse_args()
    vqa_interim(args.dir_vqa)
