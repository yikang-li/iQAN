import json
import sys
import os.path as osp


if __name__ == '__main__':

	json_file = sys.argv[1]
	print('Transferring the JSON file: {}'.format(json_file))
	assert osp.isfile(json_file), 'File Not Found: {}'.format(json_file)
	with open(json_file, 'r') as f:
		results = json.load(f)
	print('Total {} items loaded'.format(len(results)))

	output_file = osp.splitext(json_file)[0] + '.txt'
	print('Writing to TXT file to: {}...'.format(output_file),)

	with open(output_file, 'wt') as f:
		for id, item in enumerate(results):
			gt_question = ' '.join(item['readable_result']['gt_question'])
			generated_questions = []
			for q_id, q_item in enumerate(item['readable_result']['augmented_qa']):
				generated_question = ' '.join(q_item[0])
				if generated_question in generated_questions:
					continue
				generated_questions.append(generated_question)
				f.write(str(id) + '\t') # same image corresponds to the same ID
				f.write('1\t' if q_item[1] == item['readable_result']['gt_answer'] else '0\t')
				f.write(str(item['numeric_result']['augmented_qa'][q_id][2]) + '\t')
				f.write(gt_question + '\t') # the groundtruth question
				f.write(generated_question + '\n') # the generated question
		if (id + 1) % 1000 is 0:
			print('{} item processed'.format(id+1))
